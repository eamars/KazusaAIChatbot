"""Independent branch cognition that emits complete immutable bids."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    ActionAffordanceV2,
    ActionBidV2,
    BranchDefinition,
    CognitionCoreServicesV2,
    CognitionEvidenceV2,
    GoalBidDraftV2,
    ResolverAffordanceV2,
)


GOAL_COGNITION_PROMPT = '''You are one independent goal cognition branch.
Use only the supplied semantic context, evidence handles, and capability handles.
Return one complete bid draft. Do not mutate state, invent evidence, or author
the final route decision. A bid may request speech, evidence, action, deferral,
or silence. Use an empty target list when unsupported. Always provide at least
one bounded expected consequence for a complete bid.

# Output Format
Return exactly one JSON object with exactly these required fields:
intention, desired_outcome, concrete_detail, reason, target_role_handles,
evidence_handles, expected_consequences, confidence, and requested_route.
The four prose fields and confidence are strings. target_role_handles and
evidence_handles are arrays of strings; expected_consequences is a non-empty
array of strings. requested_route is one of speech, evidence, action, deferral,
or silence. Add requested_action_handle or
requested_resolver_handle only when that capability is declared by the bid.
Do not emit target_roles, role_handles, semantic_text, action details, numeric
confidence, or any other field.
'''


async def run_goal_cognition(
    definition: BranchDefinition,
    goal_ref: Mapping[str, Any],
    semantic_context: Mapping[str, Any],
    evidence: Sequence[CognitionEvidenceV2],
    action_affordances: Sequence[ActionAffordanceV2],
    resolver_affordances: Sequence[ResolverAffordanceV2],
    services: CognitionCoreServicesV2,
) -> ActionBidV2:
    """Run one goal branch and map its draft to a complete deterministic bid."""

    evidence_handles = [row["evidence_handle"] for row in evidence]
    action_handles = {
        f"a{index}": affordance
        for index, affordance in enumerate(action_affordances, start=1)
    }
    resolver_handles = {
        f"r{index}": affordance
        for index, affordance in enumerate(resolver_affordances, start=1)
    }
    role_bindings = semantic_context.get("_role_bindings", {})
    if not isinstance(role_bindings, Mapping):
        role_bindings = {}
    role_summaries = semantic_context.get("role_summaries", {})
    if not isinstance(role_summaries, Mapping):
        role_summaries = {}
    prompt_context = {
        key: value
        for key, value in semantic_context.items()
        if not key.startswith("_")
    }
    prompt_payload = {
        "branch": {
            "goal_kind": definition.goal_kind,
            "action_tendencies": list(definition.action_tendencies),
        },
        "goal": semantic_context.get(
            "goal_projection",
            {"goal_kind": definition.goal_kind, "lifecycle": "active"},
        ),
        "semantic_context": prompt_context,
        "evidence": [
            {
                "handle": row["evidence_handle"],
                "semantic_text": row["semantic_text"],
            }
            for row in evidence
        ],
        "action_handles": {
            handle: _affordance_prompt_value(value)
            for handle, value in action_handles.items()
        },
        "resolver_handles": {
            handle: _resolver_prompt_value(value)
            for handle, value in resolver_handles.items()
        },
        "role_handles": sorted(role_summaries),
        "role_summaries": dict(role_summaries),
    }
    prompt_text = json.dumps(prompt_payload, ensure_ascii=False, sort_keys=True)
    if len(prompt_text) > 24000:
        raise ValueError("goal cognition prompt exceeds the contract cap")
    response = await services.llm.ainvoke(
        [
            SystemMessage(content=GOAL_COGNITION_PROMPT),
            HumanMessage(content=prompt_text),
        ],
        config=services.goal_cognition_config,
    )
    parsed = services.parse_json(response.content)
    draft = validate_goal_bid_draft(
        parsed,
        evidence_handles=set(evidence_handles),
        role_handles=set(role_bindings),
        action_handles=set(action_handles),
        resolver_handles=set(resolver_handles),
    )
    target_roles = [dict(role_bindings[handle]) for handle in draft["target_role_handles"]]
    bid: ActionBidV2 = {
        "branch_id": definition.branch_id,
        "goal_ref": dict(goal_ref),
        "intention": draft["intention"],
        "desired_outcome": draft["desired_outcome"],
        "concrete_detail": draft["concrete_detail"],
        "reason": draft["reason"],
        "target_roles": target_roles,
        "evidence_handles": list(draft["evidence_handles"]),
        "expected_consequences": list(draft["expected_consequences"]),
        "confidence": draft["confidence"],
        "requested_route": draft["requested_route"],
    }
    if "requested_action_handle" in draft:
        bid["requested_action_kind"] = action_handles[
            draft["requested_action_handle"]
        ]["action_kind"]
    if "requested_resolver_handle" in draft:
        bid["requested_resolver_capability"] = resolver_handles[
            draft["requested_resolver_handle"]
        ]["capability"]
    return bid


def validate_goal_bid_draft(
    parsed: object,
    *,
    evidence_handles: set[str],
    role_handles: set[str],
    action_handles: set[str],
    resolver_handles: set[str],
) -> GoalBidDraftV2:
    """Validate model-owned fields before any complete bid is constructed."""

    if not isinstance(parsed, Mapping):
        raise ValueError("goal bid draft must be an object")
    required = {
        "intention",
        "desired_outcome",
        "concrete_detail",
        "reason",
        "target_role_handles",
        "evidence_handles",
        "expected_consequences",
        "confidence",
        "requested_route",
    }
    optional = {"requested_action_handle", "requested_resolver_handle"}
    if set(parsed).difference(required | optional) or not required.issubset(parsed):
        raise ValueError("goal bid draft fields are not exact")
    for field_name in ("intention", "desired_outcome", "concrete_detail", "reason"):
        _bounded_text(parsed[field_name], field_name, 500)
    _bounded_text(parsed["confidence"], "confidence", 40)
    if parsed["requested_route"] not in {
        "speech",
        "evidence",
        "action",
        "deferral",
        "silence",
    }:
        raise ValueError("goal bid route is invalid")
    target_roles = _handles(parsed["target_role_handles"], role_handles, "role")
    cited_evidence = _handles(parsed["evidence_handles"], evidence_handles, "evidence")
    consequences = parsed["expected_consequences"]
    if not isinstance(consequences, list) or not 1 <= len(consequences) <= 8:
        raise ValueError("goal bid consequences are invalid")
    for consequence in consequences:
        _bounded_text(consequence, "consequence", 240)
    if "requested_action_handle" in parsed and parsed["requested_action_handle"] not in action_handles:
        raise ValueError("goal bid action handle is unavailable")
    if "requested_resolver_handle" in parsed and parsed["requested_resolver_handle"] not in resolver_handles:
        raise ValueError("goal bid resolver handle is unavailable")
    result = dict(parsed)
    result["target_role_handles"] = target_roles
    result["evidence_handles"] = cited_evidence
    result["expected_consequences"] = consequences
    return result  # type: ignore[return-value]


def _handles(value: Any, allowed: set[str], label: str) -> list[str]:
    """Validate a duplicate-free bounded handle partition."""

    if not isinstance(value, list) or len(value) > 8:
        raise ValueError(f"{label} handles are invalid")
    if len(value) != len(set(value)) or any(handle not in allowed for handle in value):
        raise ValueError(f"{label} handles are not permitted")
    return list(value)


def _bounded_text(value: Any, label: str, maximum: int) -> None:
    """Validate bounded model-owned prose."""

    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{label} is invalid")


def _affordance_prompt_value(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project action affordance fields without operational handlers."""

    return {
        "action_kind": value["action_kind"],
        "capability": value["capability"],
        "permission": value["permission"],
    }


def _resolver_prompt_value(value: Mapping[str, Any]) -> dict[str, Any]:
    """Project resolver availability without executable capability details."""

    return {
        "capability": value["capability"],
        "semantic_capability": value["semantic_capability"],
        "availability": value["availability"],
    }
