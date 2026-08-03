"""Public V2 text and terminal visual surface planning facades."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.cognition_episode import project_model_visible_percepts
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionExecutionError,
    TextSurfaceInputV2,
    TextSurfaceOutputV2,
    TextSurfaceServicesV2,
    VisualSurfaceOutputV2,
    VisualSurfaceServicesV2,
    validate_text_surface_input,
    validate_text_surface_output,
    validate_visual_surface_output,
)
from kazusa_ai_chatbot.cognition_core_v2.surface_stages import (
    run_dialog_compliance_repair_stage,
    run_content_plan_stage,
    run_preference_stage,
    run_visual_stage,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    validate_prompt_projection,
)
from kazusa_ai_chatbot.llm_tracing import failure_capsule


async def run_text_surface_planning(
    input_payload: TextSurfaceInputV2,
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
    input_payload: TextSurfaceInputV2,
    services: TextSurfaceServicesV2,
    *,
    session: failure_capsule.FailureCapsuleSession | None,
) -> TextSurfaceOutputV2:
    """Run two bounded text-surface stages after cognition is committed."""

    payload = validate_text_surface_input(input_payload)
    stage_payload = _project_surface_payload(payload)
    validate_prompt_projection(stage_payload)
    content_payload = dict(stage_payload)
    content_payload["character_expression_context"] = dict(
        payload["character_expression_context"]
    )
    validate_prompt_projection(content_payload)
    content_result, preference = await asyncio.gather(
        run_content_plan_stage(content_payload, services),
        run_preference_stage(stage_payload, services),
        return_exceptions=True,
    )
    stage_results = (content_result, preference)
    for stage_result in stage_results:
        if (
            isinstance(stage_result, BaseException)
            and not isinstance(stage_result, CognitionExecutionError)
        ):
            raise stage_result
    if any(
        isinstance(stage_result, CognitionExecutionError)
        for stage_result in stage_results
    ):
        failed_stages = []
        if isinstance(content_result, CognitionExecutionError):
            failed_stages.append("content_plan")
        if isinstance(preference, CognitionExecutionError):
            failed_stages.append("preference")
        failure_capsule.mark_failure(
            session,
            failure_kind="degraded_surface",
            stage_name="run_text_surface_planning",
            details={"failed_stages": failed_stages},
        )
        degraded_output = build_degraded_text_surface(payload)
        return degraded_output
    content_plan, content_requirements, delivery_profile = content_result
    visible_boundaries, addressee_plan = preference
    output: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": content_plan,
        "content_requirements": content_requirements,
        "visible_boundaries": visible_boundaries,
        "addressee_plan": addressee_plan,
        "delivery_profile": delivery_profile,
        "selected_surface_intent": payload["intention"]["intention"],
        "permitted_action_results": [
            {
                **row,
                "target_roles": [
                    dict(role) for role in row["target_roles"]
                ],
            }
            for row in payload["permitted_action_results"]
        ],
    }
    if "runtime_capability_limits" in payload:
        output["runtime_capability_limits"] = list(
            payload["runtime_capability_limits"]
        )
    if "resolver_result" in payload:
        output["resolver_result"] = dict(payload["resolver_result"])
    validated_output = validate_text_surface_output(output)
    return validated_output


def build_degraded_text_surface(
    input_payload: TextSurfaceInputV2,
) -> TextSurfaceOutputV2:
    """Project a valid neutral text surface from canonical cognition truth.

    Args:
        input_payload: Validated V2 cognition-to-surface contract.

    Returns:
        A validated surface that preserves selected intent, action truth, and
        runtime capability limits without model-authored additions.
    """

    payload = validate_text_surface_input(input_payload)
    selected_intention = payload["intention"]["intention"]
    output: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": selected_intention,
        "content_requirements": [
            "表达已选择的回应意图，并保持当前事实、角色方向和能力结果原义。",
        ],
        "visible_boundaries": [],
        "addressee_plan": [],
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
    }
    if "runtime_capability_limits" in payload:
        output["runtime_capability_limits"] = list(
            payload["runtime_capability_limits"]
        )
    if "resolver_result" in payload:
        output["resolver_result"] = dict(payload["resolver_result"])
    validated_output = validate_text_surface_output(output)
    return validated_output


async def repair_text_surface_planning(
    input_payload: TextSurfaceInputV2,
    verified_hard_issues: list[str],
    services: TextSurfaceServicesV2,
) -> TextSurfaceOutputV2:
    """Repair text planning with exact multi-argument failure capture."""

    capsule_input = {
        "input_payload": input_payload,
        "verified_hard_issues": verified_hard_issues,
    }
    session = failure_capsule.begin_failure_capsule(
        trace_id=llm_tracing.current_trace_id(),
        entrypoint="repair_text_surface_planning",
        input_payload=capsule_input,
    )
    try:
        output = await _repair_text_surface_planning(
            input_payload,
            verified_hard_issues,
            services,
        )
    except Exception as exc:
        failure_capsule.mark_failure(
            session,
            failure_kind="terminal_failure",
            stage_name="repair_text_surface_planning",
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


async def _repair_text_surface_planning(
    input_payload: TextSurfaceInputV2,
    verified_hard_issues: list[str],
    services: TextSurfaceServicesV2,
) -> TextSurfaceOutputV2:
    """Replace every producer-owned field from canonical cognition truth.

    Args:
        input_payload: Canonical cognition-owned surface input.
        verified_hard_issues: Bounded verifier findings to resolve.
        services: Configured text-surface model and route settings.

    Returns:
        A validated replacement surface derived without rejected candidates.
    """

    payload = validate_text_surface_input(input_payload)
    if (
        not isinstance(verified_hard_issues, list)
        or not 1 <= len(verified_hard_issues) <= 8
        or len(verified_hard_issues) != len(set(verified_hard_issues))
    ):
        raise ValueError("verified dialog hard issues are invalid")
    if any(
        not isinstance(issue, str)
        or not issue.strip()
        or len(issue) > 300
        for issue in verified_hard_issues
    ):
        raise ValueError("verified dialog hard issue text is invalid")

    stage_payload = _project_surface_payload(payload)
    stage_payload["character_expression_context"] = dict(
        payload["character_expression_context"]
    )
    stage_payload["dialog_compliance_repair"] = {
        "verified_hard_issues": list(verified_hard_issues),
    }
    validate_prompt_projection(stage_payload)
    replacement = await run_dialog_compliance_repair_stage(
        stage_payload,
        services,
    )
    output: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": replacement["content_plan"],
        "content_requirements": list(
            replacement["content_requirements"]
        ),
        "visible_boundaries": list(replacement["visible_boundaries"]),
        "addressee_plan": list(replacement["addressee_plan"]),
        "delivery_profile": dict(replacement["delivery_profile"]),
        "selected_surface_intent": payload["intention"]["intention"],
        "permitted_action_results": [
            {
                **row,
                "target_roles": [
                    dict(role) for role in row["target_roles"]
                ],
            }
            for row in payload["permitted_action_results"]
        ],
    }
    if "runtime_capability_limits" in payload:
        output["runtime_capability_limits"] = list(
            payload["runtime_capability_limits"]
        )
    if "resolver_result" in payload:
        output["resolver_result"] = dict(payload["resolver_result"])
    validated_output = validate_text_surface_output(output)
    return validated_output


async def run_visual_surface_planning(
    input_payload: TextSurfaceInputV2,
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
    input_payload: TextSurfaceInputV2,
    services: VisualSurfaceServicesV2,
) -> VisualSurfaceOutputV2:
    """Run the independent terminal visual-directive stage."""

    payload = validate_text_surface_input(input_payload)
    stage_payload = _project_surface_payload(payload)
    stage_payload["visual_character_context"] = payload[
        "visual_character_context"
    ]
    validate_prompt_projection(stage_payload)
    visual_directives = await run_visual_stage(stage_payload, services)
    output: VisualSurfaceOutputV2 = {
        "schema_version": "visual_surface_output.v2",
        "visual_directives": visual_directives,
        "selected_surface_intent": payload["intention"]["intention"],
    }
    validated_output = validate_visual_surface_output(output)
    return validated_output


def _project_surface_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Remove persistent/private fields before any surface stage sees input."""

    intention = payload["intention"]
    projected_intention = {
        "route": intention["route"],
        "intention": intention["intention"],
        "reason": intention["reason"],
    }
    result: dict[str, Any] = {
        "episode": _project_episode(payload["episode"]),
        "intention": projected_intention,
        "goal_resolution": payload["goal_resolution"],
        "supporting_bids": payload["supporting_bids"],
        "expression_policy": payload["expression_policy"],
        "semantic_affect": payload["semantic_affect"],
        "permitted_action_results": _project_action_results_for_prompt(
            payload["permitted_action_results"]
        ),
        "interaction_style_context": payload["interaction_style_context"],
    }
    if "runtime_capability_limits" in payload:
        result["runtime_capability_limits"] = list(
            payload["runtime_capability_limits"]
        )
    if "resolver_result" in payload:
        result["resolver_result"] = dict(payload["resolver_result"])
    if "primary_bid" in payload:
        result["primary_bid"] = payload["primary_bid"]
    if "semantic_relationship" in payload:
        result["semantic_relationship"] = payload["semantic_relationship"]
    if "relational_willingness" in payload:
        result["relational_willingness"] = dict(
            payload["relational_willingness"]
        )
    return result


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
