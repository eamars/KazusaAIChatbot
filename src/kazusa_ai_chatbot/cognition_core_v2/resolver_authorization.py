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


RESOLVER_AUTHORIZATION_PROMPT = '''You are the focused semantic authorization
boundary for proposed resolver evidence work. The planner has already proposed
candidates. For every candidate, decide only whether its evidence need remains
unresolved, materially advances its cited admitted bid, and fits the supplied
resolver capability.

Current evidence and prior resolver context are authoritative. Authorize a
candidate when relevant evidence is genuinely missing and the capability can
retrieve or resolve it. Reject a candidate when current evidence already
satisfies the proposed need, when the proposal repeats an earlier need using
different wording, or when the prior resolver context shows that the same need
cannot usefully proceed. A previous successful observation does not by itself
block a distinct or materially narrower follow-up whose required evidence is
still absent.

Judge unresolved evidence need and capability fit only. Do not rewrite a
request, choose final dialogue, judge character willingness or writing quality,
or invent another capability.

# Output Format
Return exactly one JSON object with exactly one field named decisions.
decisions must be one JSON object whose keys are exactly the supplied candidate
handles and whose values are booleans. true authorizes that candidate and false
rejects it. Do not omit or invent candidates. Return JSON only.
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
