"""Single-pass, caller-bound Cognition V3 stage flow.

This module owns the phase-one model boundary.  It intentionally does not
share the historical branch, bid, or repair orchestration helpers.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot import llm_tracing
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
from kazusa_ai_chatbot.cognition_shared.state_models import validate_cognition_state
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    RELATIONSHIP_AXIS_FIELDS,
    project_affect,
)
from kazusa_ai_chatbot.cognition_shared.state_reducers import (
    apply_relationship_maintenance,
    apply_state_update,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.utils import parse_llm_json_output

logger = logging.getLogger(__name__)


class CanonicalContractError(ValueError):
    """A mechanically unusable single-pass model product."""


_STAGE_INSTRUCTIONS = {
    "A1": "Judge the current observation with the three world-facing appraisal families.",
    "A2": "Judge the accepted meaning with relationship, moral, and existential families.",
    "G": "Choose one meaningful active-character goal and relational willingness.",
    "P": "Choose the active-character response goal and only supplied capabilities.",
}

_COGNITION_STAGE_SEQUENCE = {"A1": 0, "A2": 1, "G": 2, "P": 3}
_COGNITION_TRACE_FIELDS = {
    "A1": ("event_agency", "goal_threat_outcome", "epistemic_comparison_memory"),
    "A2": ("relationship_social", "moral_identity", "existential_drive"),
    "G": ("active_character_goal", "relational_willingness"),
    "P": ("goal_resolution", "response_goal", "action_requests", "resolver_requests"),
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


def _next_transaction_timestamp(value: str) -> str:
    """Advance one persisted UTC version deterministically."""

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CanonicalContractError("canonical state timestamp is invalid") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return (
        (parsed + timedelta(microseconds=1))
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _transaction_timing(
    state: Mapping[str, object],
    episode: Mapping[str, object],
) -> tuple[int, str, str]:
    """Derive elapsed lifecycle time and mutation date from the episode."""

    raw_episode_time = episode.get("created_at")
    raw_state_time = state.get("updated_at")
    if not isinstance(raw_episode_time, str) or not isinstance(raw_state_time, str):
        raise CanonicalContractError("canonical transaction timestamps are invalid")
    try:
        episode_time = datetime.fromisoformat(raw_episode_time.replace("Z", "+00:00"))
        state_time = datetime.fromisoformat(raw_state_time.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CanonicalContractError("canonical transaction timestamps are invalid") from exc
    if episode_time.tzinfo is None:
        episode_time = episode_time.replace(tzinfo=timezone.utc)
    if state_time.tzinfo is None:
        state_time = state_time.replace(tzinfo=timezone.utc)
    elapsed_seconds = max(0, int((episode_time - state_time).total_seconds()))
    mutation_anchor = max(episode_time, state_time).astimezone(timezone.utc)
    mutation_time = _next_transaction_timestamp(
        mutation_anchor.isoformat().replace("+00:00", "Z")
    )
    return elapsed_seconds, mutation_time, episode_time.date().isoformat()


def _typed_transaction_facts(value: object) -> list[tuple[str, Mapping[str, object]]]:
    """Convert caller-owned producer/fact rows to the reducer input shape."""

    if value is None:
        return []
    if not isinstance(value, list):
        raise CanonicalContractError("canonical direct facts must be an array")
    result: list[tuple[str, Mapping[str, object]]] = []
    for row in value:
        if not isinstance(row, Mapping):
            raise CanonicalContractError("canonical direct fact row is invalid")
        producer = row.get("producer")
        fact = row.get("fact")
        if not isinstance(producer, str) or not producer.strip():
            raise CanonicalContractError("canonical direct fact producer is invalid")
        if not isinstance(fact, Mapping):
            fact = {
                key: item
                for key, item in row.items()
                if key != "producer"
            }
        result.append((producer, dict(fact)))
    return result


def _prepare_state_transaction(
    payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
    """Apply trusted lifecycle inputs before semantic appraisal binding."""

    current_state = validate_cognition_state(payload["mutable_state"])
    persisted_base = payload.get("_persisted_base_state")
    original = (
        validate_cognition_state(persisted_base)
        if isinstance(persisted_base, Mapping)
        else current_state
    )
    episode = payload.get("episode")
    if not isinstance(episode, Mapping):
        raise CanonicalContractError("canonical episode is invalid")
    direct_facts = _typed_transaction_facts(payload.get("direct_facts", []))
    elapsed_seconds, updated_at, interaction_date = _transaction_timing(
        current_state,
        episode,
    )
    evolved = apply_state_update(
        current_state,
        direct_facts=direct_facts,
        elapsed_seconds=elapsed_seconds,
        updated_at=updated_at,
        character_constraints=(
            payload.get("character_constraints")
            if isinstance(payload.get("character_constraints"), Mapping)
            else None
        ),
        relationship_context=(
            payload.get("relationship_context")
            if isinstance(payload.get("relationship_context"), Mapping)
            else None
        ),
    )
    payload["_transaction_elapsed_seconds"] = elapsed_seconds
    payload["_transaction_interaction_date"] = interaction_date
    payload["_transaction_direct_facts"] = [
        {"producer": producer, **dict(fact)}
        for producer, fact in direct_facts
    ]
    validated = validate_cognition_state(evolved)
    transitions = [
        dict(row)
        for row in payload.get("transaction_transition_contexts", [])
        if isinstance(row, Mapping)
    ]
    return dict(original), validated, transitions


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
    try:
        response = await services.llm.ainvoke(
            messages,
            config=config,
        )
    except Exception as exc:
        await _record_cognition_trace_attempt(
            stage=stage,
            config=config,
            messages=messages,
            response_text="",
            parsed_output={},
            parse_status="provider_error",
            status="failed",
            started=started,
            validation_error=str(exc),
        )
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
            "raw_output": "",
            "parsed_output": None,
            "status": "provider_error",
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        })
        raise
    raw_content = getattr(response, "content", "")
    try:
        parsed = parse_llm_json_output(raw_content, deterministic_only=True)
        if not isinstance(parsed, dict) or not parsed:
            raise CanonicalContractError(f"{stage} returned no usable JSON object")
    except (TypeError, ValueError) as exc:
        await _record_cognition_trace_attempt(
            stage=stage,
            config=config,
            messages=messages,
            response_text=str(raw_content),
            parsed_output=None,
            parse_status="contract_error",
            status="failed",
            started=started,
            validation_error=str(exc),
        )
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
    await _record_cognition_trace_attempt(
        stage=stage,
        config=config,
        messages=messages,
        response_text=str(raw_content),
        parsed_output=parsed,
        parse_status="succeeded",
        status="succeeded",
        started=started,
        validation_error="",
    )
    return parsed


async def _record_cognition_trace_attempt(
    *,
    stage: str,
    config: LLMCallConfig,
    messages: list[SystemMessage | HumanMessage],
    response_text: str,
    parsed_output: object,
    parse_status: str,
    status: str,
    started: float,
    validation_error: str,
) -> None:
    """Persist one cognition attempt without affecting semantic execution."""

    try:
        await llm_tracing.record_llm_trace_step(
            trace_id=llm_tracing.current_trace_id(),
            stage_name=f"cognition_core_v3.{stage}",
            route_name=config.route_name,
            model_name=config.model,
            messages=messages,
            response_text=response_text,
            parsed_output=parsed_output,
            parse_status=parse_status,
            status=status,
            duration_ms=max(0, int((time.monotonic() - started) * 1000)),
            output_state_fields=_COGNITION_TRACE_FIELDS[stage],
            sequence=_COGNITION_STAGE_SEQUENCE[stage],
            call_config=config,
            attempt_index=1,
            validation_error=validation_error,
            attempt_started_at=started,
        )
    except Exception as exc:
        logger.warning(
            "Cognition trace step write failed: %s",
            exc.__class__.__name__,
        )


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
    """Run the complete canonical chain under its configured deadline."""

    return await asyncio.wait_for(
        _run_cognition(input_payload, services),
        timeout=services.turn_deadline_seconds,
    )


async def _run_cognition(
    input_payload: Mapping[str, object], services: CognitionChainServicesV3
) -> dict[str, object]:
    """Run exactly A1, A2, G, and caller-selected P once each."""

    validated = _validate_canonical_input(input_payload)
    original_state, transaction_state, transaction_transitions = (
        _prepare_state_transaction(validated)
    )
    validated["mutable_state"] = transaction_state
    validated["_transaction_transition_contexts"] = transaction_transitions
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
        character_affect_context=validated.get(
            "character_affect_context", []
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
    binding_metadata: dict[str, object] = {}
    replacement_state, transition_contexts, binding_receipts, cause_provenance = bind_axis_changes(
        validated,
        appraisals,
        goal=goal.__dict__,
        willingness=willingness,
        goal_resolution=plan.goal_resolution,
        action_requests=plan.action_requests,
        resolver_requests=plan.resolver_requests,
        binding_metadata=binding_metadata,
    )
    if validated["state_scope"] == "user":
        episode = validated["episode"]
        if not isinstance(episode, Mapping):
            raise CanonicalContractError("canonical episode is invalid")
        relationship_deltas: list[dict[str, object]] = []
        for receipt in binding_receipts:
            if receipt.get("family") != "relationship_social":
                continue
            applied_targets = receipt.get("applied_targets", [])
            if not isinstance(applied_targets, list):
                continue
            for applied in applied_targets:
                if not isinstance(applied, Mapping):
                    continue
                target_path = applied.get("target_path")
                applied_delta = applied.get("applied_delta")
                if (
                    isinstance(target_path, str)
                    and target_path.startswith("relationship.")
                    and isinstance(applied_delta, int)
                    and not isinstance(applied_delta, bool)
                ):
                    relationship_deltas.append({
                        "duplicate_disposition": "unique",
                        "target_path": target_path,
                        "relationship_axis": receipt.get("axis"),
                        "applied_delta": applied_delta,
                    })
        replacement_state = apply_relationship_maintenance(
            replacement_state,
            source_episode_id=str(episode["episode_id"]),
            interaction_date_utc=str(validated["_transaction_interaction_date"]),
            elapsed_seconds=int(validated["_transaction_elapsed_seconds"]),
            accepted_relationship_deltas=relationship_deltas,
            trusted_facts=tuple(
                row for row in validated.get("_transaction_direct_facts", [])
                if isinstance(row, Mapping)
            ),
        )
        replacement_state = validate_cognition_state(replacement_state)
    derived_activations = derive_persistent_emotion_activations(
        replacement_state,
        updated_at=str(replacement_state.get("updated_at", "")),
        character_constraints=validated.get("character_constraints"),
        relationship_context=validated.get("relationship_context"),
        transition_contexts=transition_contexts,
    )
    replacement_state["affect_activations"] = derived_activations
    replacement_state = validate_cognition_state(replacement_state)
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
        "expected_previous_state": original_state,
        "original_persisted_state": original_state,
        "replacement_state": replacement_state,
        "transition_contexts": transition_contexts,
        "binding_receipts": binding_receipts,
        "capacity_deferred": [
            dict(row)
            for row in binding_receipts
            if row.get("disposition") == "capacity_deferred"
        ],
    }
    if "continuation_goal_ref" in binding_metadata:
        output["state_projection"]["continuation_goal_ref"] = dict(
            binding_metadata["continuation_goal_ref"]
        )
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
