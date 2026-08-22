"""Single-pass, caller-bound Cognition V3 stage flow.

This module owns the phase-one model boundary.  It intentionally does not
share the historical branch, bid, or repair orchestration helpers.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from dataclasses import replace

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.cognition_core_v3.appraisal import (
    bind_axis_changes,
    validate_canonical_appraisal,
)
from kazusa_ai_chatbot.cognition_core_v3.contracts import (
    CANONICAL_A1_FAMILIES,
    CANONICAL_A2_FAMILIES,
    CANONICAL_COGNITION_OUTPUT_SCHEMA,
    CanonicalAppraisal,
    CanonicalCognitionOutput,
    CanonicalGoal,
    CanonicalResponsePlan,
    CognitionChainServicesV3,
    validate_canonical_cognition_output,
)
from kazusa_ai_chatbot.cognition_core_v3.diagnostics import (
    bind_protected_chain_records,  # noqa: F401
    record_protected_chain_record,
    reset_protected_chain_records,  # noqa: F401
    snapshot_protected_chain_records,  # noqa: F401
)
from kazusa_ai_chatbot.cognition_core_v3.prompt import (
    build_canonical_appraisal_question,
    build_canonical_goal_question,
    build_canonical_plan_question,
    build_canonical_turn_workspace,
)
from kazusa_ai_chatbot.cognition_shared.contracts import (
    GOAL_RESOLUTION_VALUES,
    SELF_COGNITION_RESPONSE_DECISION_VALUES,
    is_targetless_group_self_cognition_episode,
)
from kazusa_ai_chatbot.cognition_shared.emotion_derivation import (
    derive_persistent_emotion_activations,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    RELATIONSHIP_AXIS_FIELDS,
    project_affect,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output


class CanonicalContractError(ValueError):
    """A mechanically unusable single-pass model product."""


_STAGE_INSTRUCTIONS = {
    "A1": "Judge the current observation with the three world-facing appraisal families.",
    "A2": "Judge the accepted meaning with relationship, moral, and existential families.",
    "G": "Choose one meaningful active-character goal and relational willingness.",
    "P": "Choose the active-character response goal and only supplied capabilities.",
}


def _validate_canonical_input(value: object) -> dict[str, object]:
    """Validate the single caller-owned input envelope without schema branching."""

    if not isinstance(value, Mapping):
        raise CanonicalContractError("canonical cognition input must be an object")
    required = {
        "episode", "scene_context", "evidence", "mutable_state",
        "state_scope", "character_constraints", "character_identity_context",
        "available_actions", "available_resolver_capabilities",
    }
    missing = required - set(value)
    if missing:
        raise CanonicalContractError(f"canonical cognition input missing {sorted(missing)}")
    if not isinstance(value["mutable_state"], Mapping):
        raise CanonicalContractError("canonical mutable state must be an object")
    if not isinstance(value["evidence"], list):
        raise CanonicalContractError("canonical evidence must be an array")
    if value["state_scope"] not in {"user", "character"}:
        raise CanonicalContractError("canonical state scope is invalid")
    return dict(value)


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


async def _call_once(
    *, services: CognitionChainServicesV3, stage: str, packet: dict[str, object]
) -> dict[str, object]:
    """Make one direct provider call and parse only deterministic JSON."""

    config = replace(
        services.chain_lane,
        stage_name=f"cognition_core_v3.{stage}",
        output_mode="text",
    )
    started = time.monotonic()
    messages = [
        SystemMessage(content=(
            f"{_STAGE_INSTRUCTIONS[stage]} Return exactly the JSON object "
            "described by output_contract. Never emit internal identifiers, "
            "handles, paths, or evidence references."
        )),
        HumanMessage(content=_json(packet)),
    ]
    response = await services.llm.ainvoke(
        messages,
        config=config,
    )
    raw_content = getattr(response, "content", "")
    try:
        parsed = parse_llm_json_output(raw_content, deterministic_only=True)
        if not isinstance(parsed, dict) or not parsed:
            raise CanonicalContractError(f"{stage} returned no usable JSON object")
    except (TypeError, ValueError):
        record_protected_chain_record({
            "stage": stage,
            "config": {
                "route_name": config.route_name,
                "model": config.model,
                "stage_name": config.stage_name,
            },
            "messages": [
                {"role": "system", "content": messages[0].content},
                {"role": "human", "content": messages[1].content},
            ],
            "raw_output": raw_content,
            "parsed_output": None,
            "status": "contract_fault",
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        })
        raise
    record_protected_chain_record({
        "stage": stage,
        "config": {
            "route_name": config.route_name,
            "model": config.model,
            "stage_name": config.stage_name,
        },
        "messages": [
            {"role": "system", "content": messages[0].content},
            {"role": "human", "content": messages[1].content},
        ],
        "raw_output": raw_content,
        "parsed_output": parsed,
        "status": "parsed",
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
    })
    return parsed


def _bounded_text(value: object, field: str, maximum: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise CanonicalContractError(f"{field} must be bounded non-empty text")
    return value.strip()


def _appraisal_summary(
    appraisals: tuple[CanonicalAppraisal, ...],
) -> list[dict[str, object]]:
    return [
        {
            "family": item.family,
            "applicable": item.applicable,
            "semantic_summary": item.semantic_summary,
            "cause_summary": item.cause_summary,
        }
        for item in appraisals
    ]


def _validate_goal(raw: object) -> tuple[CanonicalGoal, dict[str, object]]:
    if not isinstance(raw, dict) or set(raw) != {"active_character_goal", "relational_willingness"}:
        raise CanonicalContractError("goal product fields are not exact")
    goal = raw["active_character_goal"]
    willingness = raw["relational_willingness"]
    if not isinstance(goal, dict) or set(goal) != {"goal_kind", "intent", "reason", "cause_summary"}:
        raise CanonicalContractError("active-character goal fields are not exact")
    if not isinstance(willingness, dict) or set(willingness) != {
        "applicable", "stance", "reason", "cause_summary"
    }:
        raise CanonicalContractError("relational willingness fields are not exact")
    if not isinstance(willingness["applicable"], bool):
        raise CanonicalContractError("relational willingness applicability is invalid")
    typed_goal = CanonicalGoal(
        goal_kind=_bounded_text(goal["goal_kind"], "goal_kind", 120),
        intent=_bounded_text(goal["intent"], "goal intent"),
        reason=_bounded_text(goal["reason"], "goal reason"),
        cause_summary=_bounded_text(goal["cause_summary"], "goal cause"),
    )
    typed_willingness = {
        "applicable": willingness["applicable"],
        "stance": _bounded_text(willingness["stance"], "willingness stance", 120),
        "reason": _bounded_text(willingness["reason"], "willingness reason"),
        "cause_summary": _bounded_text(willingness["cause_summary"], "willingness cause"),
    }
    return typed_goal, typed_willingness


def _validate_plan(raw: object, *, self_cognition: bool, capabilities: dict[str, object]) -> CanonicalResponsePlan:
    if not isinstance(raw, dict):
        raise CanonicalContractError("response plan must be an object")
    if self_cognition:
        if set(raw) != {"self_cognition_response"} or not isinstance(raw["self_cognition_response"], dict):
            raise CanonicalContractError("self-cognition plan fields are not exact")
        item = raw["self_cognition_response"]
        if set(item) != {"decision", "response_goal", "reason", "cause_summary"}:
            raise CanonicalContractError("self-cognition response fields are not exact")
        if item["decision"] not in SELF_COGNITION_RESPONSE_DECISION_VALUES:
            raise CanonicalContractError("self-cognition decision is unsupported")
        return CanonicalResponsePlan(
            goal_resolution="answerable_now",
            response_goal=_bounded_text(item["response_goal"], "self response goal"),
            action_requests=(), resolver_requests=(),
            self_cognition_response={
                "decision": item["decision"],
                "response_goal": _bounded_text(item["response_goal"], "self response goal"),
                "reason": _bounded_text(item["reason"], "self response reason"),
                "cause_summary": _bounded_text(item["cause_summary"], "self response cause"),
            },
        )
    required = {"goal_resolution", "response_goal", "action_requests", "resolver_requests"}
    if set(raw) != required or raw["goal_resolution"] not in GOAL_RESOLUTION_VALUES:
        raise CanonicalContractError("ordinary response plan fields are not exact")
    action_roster = {
        row.get("action_kind") for row in capabilities.get("actions", []) if isinstance(row, dict)
    }
    resolver_roster = {
        row.get("capability") for row in capabilities.get("resolvers", []) if isinstance(row, dict)
    }
    actions = raw["action_requests"]
    resolvers = raw["resolver_requests"]
    if not isinstance(actions, list) or not isinstance(resolvers, list):
        raise CanonicalContractError("response capability requests must be arrays")
    clean_actions: list[dict[str, object]] = []
    action_affordances = {
        row["action_kind"]: row
        for row in capabilities.get("actions", [])
        if isinstance(row, dict) and isinstance(row.get("action_kind"), str)
    }
    resolver_affordances = {
        row["capability"]: row
        for row in capabilities.get("resolvers", [])
        if isinstance(row, dict) and isinstance(row.get("capability"), str)
    }
    if len(action_affordances) != len(capabilities.get("actions", [])):
        raise CanonicalContractError("action capabilities are duplicated")
    if len(resolver_affordances) != len(capabilities.get("resolvers", [])):
        raise CanonicalContractError("resolver capabilities are duplicated")
    seen_action_kinds: set[str] = set()
    for row in actions:
        if not isinstance(row, dict) or set(row) != {"action_kind", "decision", "detail", "reason"}:
            raise CanonicalContractError("action request fields are not exact")
        action_kind = row["action_kind"]
        if not isinstance(action_kind, str) or action_kind not in action_roster:
            raise CanonicalContractError("action capability is not available")
        if action_kind in seen_action_kinds:
            raise CanonicalContractError("action capability is duplicated")
        seen_action_kinds.add(action_kind)
        affordance = action_affordances[action_kind]
        decision = _bounded_text(row["decision"], "action decision")
        decision_mode = affordance.get("decision_mode")
        allowed_decisions = affordance.get("allowed_decisions", [])
        if decision_mode == "closed" and decision not in allowed_decisions:
            raise CanonicalContractError("action decision is outside its closed affordance")
        if decision_mode == "required_text":
            pattern = affordance.get("decision_pattern", "")
            if not isinstance(pattern, str) or not re.fullmatch(pattern, decision):
                raise CanonicalContractError("action decision does not match its affordance")
        clean_actions.append({key: _bounded_text(row[key], f"action {key}") for key in row})
    max_actions = 2 if str(raw["response_goal"]).strip() else 3
    if len(clean_actions) > max_actions:
        raise CanonicalContractError("action request capacity is exceeded")
    clean_resolvers: list[dict[str, object]] = []
    seen_resolver_capabilities: set[str] = set()
    for row in resolvers:
        if not isinstance(row, dict) or set(row) != {"capability", "goal", "reason"}:
            raise CanonicalContractError("resolver request fields are not exact")
        capability = row["capability"]
        if not isinstance(capability, str) or capability not in resolver_roster:
            raise CanonicalContractError("resolver capability is not available")
        if capability in seen_resolver_capabilities:
            raise CanonicalContractError("resolver capability is duplicated")
        seen_resolver_capabilities.add(capability)
        if capability not in resolver_affordances:
            raise CanonicalContractError("resolver capability affordance is missing")
        clean_resolvers.append({key: _bounded_text(row[key], f"resolver {key}") for key in row})
    return CanonicalResponsePlan(
        goal_resolution=raw["goal_resolution"],
        response_goal=_bounded_text(raw["response_goal"], "response goal"),
        action_requests=tuple(clean_actions), resolver_requests=tuple(clean_resolvers),
    )


async def run_cognition(
    input_payload: Mapping[str, object], services: CognitionChainServicesV3
) -> dict[str, object]:
    """Run exactly A1, A2, G, and caller-selected P once each."""

    validated = _validate_canonical_input(input_payload)
    workspace = build_canonical_turn_workspace(
        episode=validated["episode"], scene_context=validated["scene_context"],
        evidence=validated["evidence"], mutable_state=validated["mutable_state"],
        character_constraints=validated["character_constraints"],
        identity_context=validated["character_identity_context"],
        continuity={
            "private": validated.get("private_continuity_context", ""),
            "dialog": validated.get("past_dialog_cognition_context", ""),
        },
        available_actions=validated["available_actions"],
        available_resolvers=validated["available_resolver_capabilities"],
        direct_facts=validated.get("direct_facts", []),
        character_operational_context=validated.get(
            "character_operational_context", {}
        ),
        relationship_context=validated.get("relationship_context", {}),
        resolver_context=validated.get("resolver_context", ""),
        resolver_progress=validated.get("resolver_goal_progress", {}),
        runtime_limits=validated.get("runtime_capability_limits", []),
        group_engagement=validated.get("group_engagement_action_context", {}),
    )
    a1_raw = await _call_once(
        services=services, stage="A1",
        packet=build_canonical_appraisal_question(workspace=workspace, stage_name="A1"),
    )
    a1 = validate_canonical_appraisal(a1_raw, families=CANONICAL_A1_FAMILIES)
    a1_summary = _appraisal_summary(a1)
    a2_raw = await _call_once(
        services=services, stage="A2",
        packet=build_canonical_appraisal_question(
            workspace=workspace,
            stage_name="A2",
            accepted_appraisal_summary=a1_summary,
        ),
    )
    a2 = validate_canonical_appraisal(a2_raw, families=CANONICAL_A2_FAMILIES)
    appraisals = (*a1, *a2)
    summaries = _appraisal_summary(appraisals)
    goal_raw = await _call_once(
        services=services, stage="G",
        packet=build_canonical_goal_question(workspace=workspace, appraisal_summary=summaries),
    )
    goal, willingness = _validate_goal(goal_raw)
    self_cognition = _is_self_cognition(validated)
    plan_raw = await _call_once(
        services=services, stage="P",
        packet=build_canonical_plan_question(
            workspace=workspace,
            goal=goal.__dict__,
            appraisal_summary=summaries,
            self_cognition=self_cognition,
        ),
    )
    plan = _validate_plan(plan_raw, self_cognition=self_cognition, capabilities=workspace["capabilities"])
    replacement_state, transition_contexts, binding_receipts, cause_provenance = bind_axis_changes(
        validated,
        appraisals,
        goal=goal.__dict__,
        willingness=willingness,
    )
    derived_activations = derive_persistent_emotion_activations(
        replacement_state,
        updated_at=str(replacement_state.get("updated_at", "")),
        character_constraints=validated.get("character_constraints"),
        relationship_context=validated.get("relationship_context"),
        transition_contexts=transition_contexts,
    )
    replacement_state["affect_activations"] = derived_activations
    affect_projection = project_affect(derived_activations, replacement_state)
    result = CanonicalCognitionOutput(
        schema_version=CANONICAL_COGNITION_OUTPUT_SCHEMA,
        appraisals=tuple(appraisals), active_character_goal=goal,
        relational_willingness=willingness, response_plan=plan,
        affect_projection=tuple(affect_projection),
        relationship_projection=_canonical_relationship_projection({
            "relationship_context": {"axes": replacement_state.get("relationship", {})}
        }),
        cause_provenance=tuple(cause_provenance),
        diagnostics={"status": "complete"},
    )
    output = result.as_dict()
    # State replacement is caller-owned.  The model never sees this carrier;
    # immediate adapters use it for the existing compare-and-replace boundary.
    output["state_projection"] = {
        "state_scope": validated["state_scope"],
        "owner_key": validated["mutable_state"].get("owner_user_id", "")
        if isinstance(validated["mutable_state"], Mapping)
        else "",
        "expected_previous_state": dict(validated["mutable_state"]),
        "replacement_state": replacement_state,
        "transition_contexts": transition_contexts,
        "binding_receipts": binding_receipts,
    }
    return dict(validate_canonical_cognition_output(output))


def _is_self_cognition(payload: Mapping[str, object]) -> bool:
    episode = payload.get("episode")
    scene = payload.get("scene_context")
    return bool(
        is_targetless_group_self_cognition_episode(payload)
        or (
        isinstance(episode, Mapping)
        and isinstance(scene, Mapping)
        and scene.get("operation") == "要求当前角色回答自己此时此刻的心理期待内容"
        )
    )


def _canonical_relationship_projection(payload: Mapping[str, object]) -> dict[str, object]:
    relationship = payload.get("relationship_context")
    if not isinstance(relationship, Mapping):
        return {"summary": "no relationship-specific context"}
    raw_axes = relationship.get("axes")
    axes = {
        name: value
        for name, value in raw_axes.items()
        if name in RELATIONSHIP_AXIS_FIELDS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    } if isinstance(raw_axes, Mapping) else {}
    return {
        "summary": "current relationship context remains caller-owned",
        "axes": axes,
    }
