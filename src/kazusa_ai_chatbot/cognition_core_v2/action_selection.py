"""Route-only model selection with deterministic availability validation."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionAffordanceV2,
    ActionBidV2,
    CognitionCoreServicesV2,
    CognitionExecutionError,
    ResolverAffordanceV2,
    ResolverCapabilityRequestV2,
    SelectedIntentionV2,
    SemanticActionRequestV2,
)


ROUTE_PROMPT = '''Select only a route from the supplied complete bid handles.
Return JSON with selected_bid_handle and route, plus action_handle or
resolver_handle only when the selected bid declares that capability. Do not
author intention, reason, targets, details, or evidence.
'''


async def select_route(
    primary_bid: ActionBidV2 | None,
    supporting_bids: Sequence[ActionBidV2],
    available_actions: Sequence[ActionAffordanceV2],
    available_resolvers: Sequence[ResolverAffordanceV2],
    services: CognitionCoreServicesV2,
) -> tuple[
    SelectedIntentionV2,
    list[SemanticActionRequestV2],
    list[ResolverCapabilityRequestV2],
]:
    """Select a route and validate it against the selected complete bid."""

    if primary_bid is None:
        return _silence_result()
    if primary_bid["requested_route"] == "action" and not any(
        affordance["action_kind"] == primary_bid.get("requested_action_kind")
        and affordance["permission"] == "allowed"
        for affordance in available_actions
    ):
        raise CognitionExecutionError("selected action is unavailable")
    if primary_bid["requested_route"] == "evidence" and not any(
        affordance["capability"]
        == primary_bid.get("requested_resolver_capability")
        and affordance["availability"] == "available"
        for affordance in available_resolvers
    ):
        raise CognitionExecutionError("selected resolver is unavailable")
    bids = [primary_bid, *supporting_bids]
    bid_handles = {
        f"b{index}": bid for index, bid in enumerate(bids, start=1)
    }
    action_handles = {
        f"a{index}": affordance
        for index, affordance in enumerate(available_actions, start=1)
    }
    resolver_handles = {
        f"r{index}": affordance
        for index, affordance in enumerate(available_resolvers, start=1)
    }
    prompt_payload = {
        "bids": {
            handle: {
                "requested_route": bid["requested_route"],
                "requested_action_kind": bid.get("requested_action_kind"),
                "requested_resolver_capability": bid.get(
                    "requested_resolver_capability"
                ),
            }
            for handle, bid in bid_handles.items()
        },
        "actions": {
            handle: {
                "action_kind": affordance["action_kind"],
                "permission": affordance["permission"],
            }
            for handle, affordance in action_handles.items()
        },
        "resolvers": {
            handle: {
                "capability": affordance["capability"],
                "availability": affordance["availability"],
            }
            for handle, affordance in resolver_handles.items()
        },
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > 12000:
        raise CognitionExecutionError("route prompt exceeds the contract cap")
    response = await services.llm.ainvoke(
        [SystemMessage(content=ROUTE_PROMPT), HumanMessage(content=prompt_text)],
        config=services.action_selection_config,
    )
    decision = _validate_route_decision(
        services.parse_json(response.content),
        bid_handles,
        action_handles,
        resolver_handles,
    )
    selected_bid = bid_handles[decision["selected_bid_handle"]]
    if decision["route"] != selected_bid["requested_route"]:
        raise CognitionExecutionError("route decision conflicts with selected bid")
    action_requests: list[SemanticActionRequestV2] = []
    resolver_requests: list[ResolverCapabilityRequestV2] = []
    if decision["route"] == "action":
        action = action_handles[decision.get("action_handle", "")]
        if (
            selected_bid.get("requested_action_kind") != action["action_kind"]
            or action["permission"] != "allowed"
        ):
            raise CognitionExecutionError("route action is unavailable or undeclared")
        action_requests.append(
            {
                "action_kind": action["action_kind"],
                "semantic_goal": selected_bid["desired_outcome"],
                "target_roles": list(selected_bid["target_roles"]),
                "evidence_handles": list(selected_bid["evidence_handles"]),
            }
        )
    elif decision["route"] == "evidence":
        resolver = resolver_handles[decision.get("resolver_handle", "")]
        if (
            selected_bid.get("requested_resolver_capability")
            != resolver["capability"]
            or resolver["availability"] != "available"
        ):
            raise CognitionExecutionError("route resolver is unavailable or undeclared")
        resolver_requests.append(
            {
                "capability": resolver["capability"],
                "semantic_goal": selected_bid["desired_outcome"],
                "evidence_handles": list(selected_bid["evidence_handles"]),
            }
        )
    intention: SelectedIntentionV2 = {
        "selected_branch_id": selected_bid["branch_id"],
        "route": decision["route"],
        "intention": selected_bid["intention"],
        "target_roles": list(selected_bid["target_roles"]),
        "reason": selected_bid["reason"],
    }
    return intention, action_requests, resolver_requests


def _validate_route_decision(
    parsed: object,
    bids: Mapping[str, ActionBidV2],
    actions: Mapping[str, ActionAffordanceV2],
    resolvers: Mapping[str, ResolverAffordanceV2],
) -> dict[str, Any]:
    """Validate route-only fields and prompt-local handle ownership."""

    if not isinstance(parsed, Mapping):
        raise ValueError("route decision must be an object")
    required = {"selected_bid_handle", "route"}
    optional = {"action_handle", "resolver_handle"}
    if set(parsed).difference(required | optional) or not required.issubset(parsed):
        raise ValueError("route decision fields are not exact")
    if parsed["selected_bid_handle"] not in bids:
        raise ValueError("route bid handle is unavailable")
    if parsed["route"] not in {
        "speech",
        "evidence",
        "action",
        "deferral",
        "silence",
    }:
        raise ValueError("route decision route is invalid")
    if "action_handle" in parsed and parsed["action_handle"] not in actions:
        raise ValueError("route action handle is unavailable")
    if "resolver_handle" in parsed and parsed["resolver_handle"] not in resolvers:
        raise ValueError("route resolver handle is unavailable")
    if parsed["route"] == "action" and "action_handle" not in parsed:
        raise ValueError("action route requires an action handle")
    if parsed["route"] == "evidence" and "resolver_handle" not in parsed:
        raise ValueError("evidence route requires a resolver handle")
    return dict(parsed)


def _silence_result() -> tuple[
    SelectedIntentionV2,
    list[SemanticActionRequestV2],
    list[ResolverCapabilityRequestV2],
]:
    """Return the frozen no-valid-bid route without an LLM call."""

    return (
        {
            "route": "silence",
            "intention": "remain silent",
            "target_roles": [],
            "reason": "no valid admitted bid",
        },
        [],
        [],
    )
