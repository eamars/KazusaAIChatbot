"""Provenance-safe workspace collapse for complete V2 action bids."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionBidV2,
    CollapsedIntentionV2,
    CognitionCoreServicesV2,
)
from kazusa_ai_chatbot.cognition_core_v2.branch_activation import (
    branch_order_key,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


COLLAPSE_PROMPT = '''Collapse complete goal bids into a prompt-local partition.
Return only JSON with primary_bid_handle, supporting_bid_handles, and
suppressed_bid_handles. Do not rewrite bid content, copy content, or invent
detail.
Every supplied bid handle must occur exactly once across the three partitions.
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
                "requested_route": bid["requested_route"],
                "confidence": bid["confidence"],
            }
            for handle, bid in handles.items()
        }
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > 24000:
        raise ValueError("workspace collapse prompt exceeds the contract cap")
    response = await services.llm.ainvoke(
        [SystemMessage(content=COLLAPSE_PROMPT), HumanMessage(content=prompt_text)],
        config=services.collapse_config,
    )
    parsed = parse_llm_json_output(response.content)
    partition = _validate_partition(parsed, set(handles))
    primary_handle = partition["primary_bid_handle"]
    primary = handles[primary_handle]
    declared_supporting = [
        handles[handle] for handle in partition["supporting_bid_handles"]
    ]
    if any(
        bid["requested_route"] != primary["requested_route"]
        for bid in declared_supporting
    ):
        raise ValueError("supporting bid route conflicts with primary")
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
