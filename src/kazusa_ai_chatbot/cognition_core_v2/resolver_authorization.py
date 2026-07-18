"""Focused semantic authorization for proposed resolver evidence work."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.action_authorization import (
    invoke_semantic_authorizer,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionBidV2,
    CognitionCoreServicesV2,
    CognitionEvidenceV2,
    CognitionExecutionError,
    ResolverAffordanceV2,
)


RESOLVER_AUTHORIZATION_PROMPT_CAP = 24000


RESOLVER_AUTHORIZATION_PROMPT = '''你负责核准规划阶段提出的证据解析工作。对每个候选项，只判断
它需要的证据是否仍未解决、是否能实质推进所引用的已接纳目标，以及是否符合所给 resolver 能力。

当前证据和已有 resolver 上下文具有最高依据。当相关证据确实缺失且所给能力能够检索或解决时，
可以核准候选项。若当前证据已经满足该需求、候选项只是换一种说法重复先前需求，或已有 resolver
上下文表明同一需求无法继续产生有效进展，则拒绝候选项。先前一次成功观察本身不妨碍不同的、或
实质上更窄且所需证据仍缺失的后续请求。

本阶段只判断未解决的证据需求与能力匹配，不改写请求、不选择最终对话，也不判断角色意愿、文笔
或虚构其他能力。

# 输出格式
只返回一个 JSON 对象，且字段必须恰好是 decisions。decisions 是一个 JSON 对象，键必须恰好
覆盖提供的 candidate handle，值必须是布尔值；true 表示核准，false 表示拒绝。候选项不得遗漏
或增添，只输出 JSON。
'''


async def authorize_resolver_requests(
    *,
    resolver_requests: Sequence[Mapping[str, str]],
    bid_handles: Mapping[str, ActionBidV2],
    evidence: Sequence[CognitionEvidenceV2],
    resolver_handles: Mapping[str, ResolverAffordanceV2],
    resolver_context: str,
    services: CognitionCoreServicesV2,
) -> list[dict[str, str]]:
    """Retain proposed resolver work whose evidence need remains useful."""

    if not resolver_requests:
        return []

    current_evidence = [
        {
            "handle": row["evidence_handle"],
            "source_kind": row["evidence_ref"]["source_kind"],
            "semantic_text": row["semantic_text"],
        }
        for row in evidence
    ]
    candidates: dict[str, dict[str, Any]] = {}
    candidate_requests: dict[str, dict[str, str]] = {}
    for index, request in enumerate(resolver_requests, start=1):
        bid_handle = request["bid_handle"]
        resolver_handle = request["resolver_handle"]
        if bid_handle not in bid_handles:
            raise CognitionExecutionError(
                "resolver authorization received an unknown bid handle"
            )
        if resolver_handle not in resolver_handles:
            raise CognitionExecutionError(
                "resolver authorization received an unknown resolver handle"
            )
        bid = bid_handles[bid_handle]
        affordance = resolver_handles[resolver_handle]
        candidate_handle = f"c{index}"
        candidate_requests[candidate_handle] = dict(request)
        candidates[candidate_handle] = {
            "capability_kind": affordance["capability"],
            "semantic_capability": affordance["semantic_capability"],
            "proposed_semantic_goal": request["semantic_goal"],
            "proposed_reason": request["reason"],
            "admitted_bid": {
                "intention": bid["intention"],
                "desired_outcome": bid["desired_outcome"],
                "concrete_detail": bid["concrete_detail"],
                "reason": bid["reason"],
                "cited_evidence_handles": list(bid["evidence_handles"]),
            },
            "current_evidence": current_evidence,
            "resolver_context": resolver_context,
        }
    prompt_text = json.dumps(
        {"candidates": candidates},
        ensure_ascii=False,
        sort_keys=True,
    )
    if len(prompt_text) > RESOLVER_AUTHORIZATION_PROMPT_CAP:
        raise CognitionExecutionError(
            "resolver-authorization prompt exceeds contract cap"
        )
    messages: list[BaseMessage] = [
        SystemMessage(content=RESOLVER_AUTHORIZATION_PROMPT),
        HumanMessage(content=prompt_text),
    ]
    decisions = await invoke_semantic_authorizer(
        services=services,
        messages=messages,
        candidate_handles=list(candidate_requests),
        stage_name="resolver_authorization",
        output_state_fields=["authorized_resolver_requests"],
    )
    authorized_handles = {
        handle for handle, authorized in decisions.items() if authorized
    }
    return [
        candidate_requests[handle]
        for handle in candidate_requests
        if handle in authorized_handles
    ]
