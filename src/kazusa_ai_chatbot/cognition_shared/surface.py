"""Canonical text and terminal visual surface planning facades."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_episode import project_model_visible_percepts
from kazusa_ai_chatbot.cognition_shared.contracts import (
    CognitionExecutionError,
    TextSurfaceInput,
    TextSurfaceOutputV2,
    TextSurfaceServicesV2,
    VisualSurfaceOutputV2,
    VisualSurfaceServicesV2,
    validate_text_surface_input_canonical,
    validate_text_surface_output,
    validate_visual_surface_output,
)
from kazusa_ai_chatbot.cognition_shared.state_projection import (
    validate_prompt_projection,
)
from kazusa_ai_chatbot.cognition_shared.surface_stages import (
    run_content_plan_stage,
    run_visual_stage,
)
from kazusa_ai_chatbot.llm_tracing import failure_capsule


async def run_text_surface_planning(
    input_payload: TextSurfaceInput,
    services: TextSurfaceServicesV2,
) -> TextSurfaceOutputV2:
    """Run text planning with failure-only protected replay capture."""

    session = failure_capsule.begin_failure_capsule(
        trace_id=llm_tracing.current_trace_id(),
        entrypoint="run_text_surface_planning",
        input_payload=input_payload,
    )
    try:
        output = await _run_text_surface_planning(
            input_payload,
            services,
            session=session,
        )
    except Exception as exc:
        failure_capsule.mark_failure(
            session,
            failure_kind="terminal_failure",
            stage_name="run_text_surface_planning",
            details={},
        )
        failure_capsule.finish_failure_capsule(
            session,
            outcome="terminal_failure",
            exception=exc,
        )
        raise

    failure_capsule.finish_failure_capsule(session, outcome=None)
    return output


async def _run_text_surface_planning(
    input_payload: TextSurfaceInput,
    services: TextSurfaceServicesV2,
    *,
    session: failure_capsule.FailureCapsuleSession | None,
) -> TextSurfaceOutputV2:
    """Run one bounded text-surface stage after cognition is committed."""

    payload = validate_text_surface_input_canonical(input_payload)
    stage_payload = _project_surface_payload(payload)
    validate_prompt_projection(stage_payload)
    content_payload = dict(stage_payload)
    content_payload["character_expression_context"] = dict(
        payload["character_expression_context"]
    )
    validate_prompt_projection(content_payload)
    try:
        content_result = await run_content_plan_stage(
            content_payload,
            services,
        )
    except CognitionExecutionError:
        failure_capsule.mark_failure(
            session,
            failure_kind="degraded_surface",
            stage_name="run_text_surface_planning",
            details={"failed_stages": ["content_plan"]},
        )
        degraded_output = build_degraded_text_surface(payload)
        return degraded_output
    (
        content_plan,
        content_requirements,
        delivery_profile,
        lexical_avoidances,
    ) = content_result
    output: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": content_plan,
        "content_requirements": content_requirements,
        "epistemic_boundary": payload[
            "subjective_expression_context"
        ]["epistemic_boundary"],
        "visible_boundaries": [],
        "addressee_plan": _deterministic_addressee_plan(payload),
        "delivery_profile": delivery_profile,
        "selected_surface_intent": payload["response_plan"].get("response_goal", ""),
        "permitted_action_results": [
            {
                **row,
                "target_roles": [
                    dict(role) for role in row["target_roles"]
                ],
            }
            for row in payload["permitted_action_results"]
        ],
        "lexical_avoidances": lexical_avoidances,
    }
    if "runtime_capability_limits" in payload:
        output["runtime_capability_limits"] = list(
            payload["runtime_capability_limits"]
        )
    if "resolver_result" in payload:
        output["resolver_result"] = dict(payload["resolver_result"])
    if "relational_willingness" in payload:
        output["relational_willingness"] = dict(
            payload["relational_willingness"]
        )
    validated_output = validate_text_surface_output(output)
    return validated_output


def build_degraded_text_surface(
    input_payload: TextSurfaceInput,
) -> TextSurfaceOutputV2:
    """Project a valid neutral text surface from canonical cognition truth.

    Args:
        input_payload: Validated V2 cognition-to-surface contract.

    Returns:
        A validated surface that preserves selected intent, action truth, and
        runtime capability limits without model-authored additions.
    """

    payload = validate_text_surface_input_canonical(input_payload)
    selected_intention = payload["response_plan"].get("response_goal", "")
    output: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": selected_intention,
        "content_requirements": [
            "表达已选择的回应意图，并保持当前事实、角色方向和能力结果原义。",
        ],
        "epistemic_boundary": payload[
            "subjective_expression_context"
        ]["epistemic_boundary"],
        "visible_boundaries": [],
        "addressee_plan": _deterministic_addressee_plan(
            payload,
        ),
        "delivery_profile": {
            "lexical_register": "自然、清楚",
            "sentence_shape": "简洁完整",
            "rhythm": "平稳",
            "hesitation": "按语义需要自然呈现",
            "punctuation": "克制清晰",
        },
        "selected_surface_intent": selected_intention,
        "permitted_action_results": [
            {
                **row,
                "target_roles": [
                    dict(role) for role in row["target_roles"]
                ],
            }
            for row in payload["permitted_action_results"]
        ],
        "lexical_avoidances": [],
    }
    if "runtime_capability_limits" in payload:
        output["runtime_capability_limits"] = list(
            payload["runtime_capability_limits"]
        )
    if "resolver_result" in payload:
        output["resolver_result"] = dict(payload["resolver_result"])
    if "relational_willingness" in payload:
        output["relational_willingness"] = dict(
            payload["relational_willingness"]
        )
    validated_output = validate_text_surface_output(output)
    return validated_output


async def run_visual_surface_planning(
    input_payload: TextSurfaceInput,
    services: VisualSurfaceServicesV2,
) -> VisualSurfaceOutputV2:
    """Run visual planning with failure-only protected replay capture."""

    session = failure_capsule.begin_failure_capsule(
        trace_id=llm_tracing.current_trace_id(),
        entrypoint="run_visual_surface_planning",
        input_payload=input_payload,
    )
    try:
        output = await _run_visual_surface_planning(input_payload, services)
    except Exception as exc:
        failure_capsule.mark_failure(
            session,
            failure_kind="terminal_failure",
            stage_name="run_visual_surface_planning",
            details={},
        )
        failure_capsule.finish_failure_capsule(
            session,
            outcome="terminal_failure",
            exception=exc,
        )
        raise

    failure_capsule.finish_failure_capsule(session, outcome=None)
    return output


async def _run_visual_surface_planning(
    input_payload: TextSurfaceInput,
    services: VisualSurfaceServicesV2,
) -> VisualSurfaceOutputV2:
    """Run the independent terminal visual-directive stage."""

    payload = validate_text_surface_input_canonical(input_payload)
    stage_payload = _project_surface_payload(payload)
    stage_payload.pop("subjective_expression_context")
    stage_payload["visual_character_context"] = payload[
        "visual_character_context"
    ]
    validate_prompt_projection(stage_payload)
    visual_directives = await run_visual_stage(stage_payload, services)
    output: VisualSurfaceOutputV2 = {
        "schema_version": "visual_surface_output.v2",
        "visual_directives": visual_directives,
        "selected_surface_intent": payload["response_plan"].get("response_goal", ""),
    }
    validated_output = validate_visual_surface_output(output)
    return validated_output


def _project_surface_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Project bounded semantic fields for the selected surface stage."""

    plan = payload["response_plan"]
    intention = {
        "route": "speech" if plan.get("response_goal") else "silence",
        "intention": plan.get("response_goal", ""),
        "reason": payload["active_character_goal"].get("reason", ""),
    }
    projected_intention = {
        "route": intention["route"],
        "intention": intention["intention"],
        "reason": intention["reason"],
    }
    # The selected operation remains a deterministic role carrier outside the
    # surface model's writable prompt projection.
    result: dict[str, Any] = {
        "episode": _project_episode(payload["episode"]),
        "intention": projected_intention,
        "active_character_goal": payload["active_character_goal"],
        "response_plan": payload["response_plan"],
        "expression_policy": payload["expression_policy"],
        "semantic_affect": payload["semantic_affect"],
        "overused_moves": list(payload["overused_moves"]),
        "permitted_action_results": _project_action_results_for_prompt(
            payload["permitted_action_results"]
        ),
        "interaction_style_context": payload["interaction_style_context"],
        "subjective_expression_context": dict(
            payload["subjective_expression_context"]
        ),
    }
    if "recent_character_dialog" in payload:
        result["recent_character_dialog"] = list(
            payload["recent_character_dialog"]
        )
    if "addressee_plan" in payload:
        result["addressee_plan"] = [
            dict(row)
            for row in payload["addressee_plan"]
        ]
    if "runtime_capability_limits" in payload:
        result["runtime_capability_limits"] = list(
            payload["runtime_capability_limits"]
        )
    if "resolver_result" in payload:
        result["resolver_result"] = _project_resolver_result_for_surface(
            payload["resolver_result"]
        )
    if "semantic_relationship" in payload:
        result["semantic_relationship"] = payload["semantic_relationship"]
    if "relational_willingness" in payload:
        result["relational_willingness"] = dict(
            payload["relational_willingness"]
        )
    return result


def _deterministic_addressee_plan(
    input_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Copy cognition-owned target identity into the public surface."""

    source = input_payload.get("addressee_plan")
    if not isinstance(source, list):
        raise ValueError("surface addressee plan must be a list")
    return [dict(row) for row in source]


def _project_action_results_for_prompt(
    action_results: object,
) -> list[dict[str, Any]]:
    """Project exact lifecycle truth without persistent target identifiers."""

    if not isinstance(action_results, list):
        raise ValueError("surface action results must be a list")
    projected: list[dict[str, Any]] = []
    for row in action_results:
        if not isinstance(row, Mapping):
            raise ValueError("surface action result must be an object")
        roles = row.get("target_roles")
        if not isinstance(roles, list):
            raise ValueError("surface action result roles must be a list")
        projected_roles = []
        for role in roles:
            if not isinstance(role, Mapping):
                raise ValueError("surface action result role must be an object")
            projected_roles.append({
                "role": role["role"],
                "entity_kind": role["entity_kind"],
            })
        projected.append({
            "action_kind": row["action_kind"],
            "status": row["status"],
            "semantic_result": row["semantic_result"],
            "target_roles": projected_roles,
        })
    return projected


def _project_resolver_result_for_surface(
    resolver_result: Mapping[str, Any],
) -> dict[str, Any]:
    """Project a source-owned resolver result without factual evidence leaks.

    Only succeeded task results with complete or partial evidence may expose
    evidence excerpts and handles to the surface model. Pending, missing,
    blocked, and unavailable/failed results are projected with those fields
    emptied so the wording owner can never treat them as completed facts.
    """

    projected = dict(resolver_result)
    evidence_state = projected.get("evidence_state")
    status = projected.get("status")
    non_factual = (
        evidence_state in {"pending", "missing", "blocked"}
        or status in {"blocked", "failed"}
    )
    if non_factual:
        projected["evidence_excerpts"] = []
        projected["evidence_handles"] = []
    return projected


def _project_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Project visible typed percepts and configured-local time for L3."""

    visible_percepts = project_model_visible_percepts(episode)
    local_time = _canonical_local_time_context(episode)
    return {
        "visible_percepts": visible_percepts,
        "local_time_context": {
            "current_local_datetime": local_time["current_local_datetime"],
            "current_local_weekday": local_time["current_local_weekday"],
        },
    }


def _canonical_local_time_context(
    episode: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Extract the canonical local-time percept for surface prompts."""

    for percept in episode["percepts"]:
        if not isinstance(percept, Mapping):
            continue
        if percept.get("percept_kind") != "local_time_context":
            continue
        content = percept.get("content")
        if not isinstance(content, Mapping):
            continue
        local_time = content.get("local_time_context")
        if isinstance(local_time, Mapping):
            return local_time
    raise ValueError("canonical episode is missing local_time_context")
