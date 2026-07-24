"""Public V2 text and terminal visual surface planning facades."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.cognition_episode import project_model_visible_percepts
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
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
    run_content_plan_stage,
    run_preference_stage,
    run_style_stage,
    run_visual_stage,
)
from kazusa_ai_chatbot.cognition_core_v2.state_projection import (
    validate_prompt_projection,
)


async def run_text_surface_planning(
    input_payload: TextSurfaceInputV2,
    services: TextSurfaceServicesV2,
) -> TextSurfaceOutputV2:
    """Run three bounded text-surface stages after cognition is committed."""

    payload = validate_text_surface_input(input_payload)
    stage_payload = _project_surface_payload(payload)
    validate_prompt_projection(stage_payload)
    style_payload = dict(stage_payload)
    style_payload["character_voice_context"] = payload[
        "character_voice_context"
    ]
    validate_prompt_projection(style_payload)
    style, content_result, preference = await asyncio.gather(
        run_style_stage(style_payload, services),
        run_content_plan_stage(stage_payload, services),
        run_preference_stage(stage_payload, services),
    )
    content_plan, content_requirements = content_result
    visible_boundaries, addressee_plan = preference
    output: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": content_plan,
        "content_requirements": content_requirements,
        "visible_boundaries": visible_boundaries,
        "addressee_plan": addressee_plan,
        "style_guidance": style,
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
    validated_output = validate_text_surface_output(output)
    return validated_output


async def run_visual_surface_planning(
    input_payload: TextSurfaceInputV2,
    services: VisualSurfaceServicesV2,
) -> VisualSurfaceOutputV2:
    """Run the independent terminal visual-directive stage."""

    payload = validate_text_surface_input(input_payload)
    stage_payload = _project_surface_payload(payload)
    stage_payload["character_voice_context"] = payload[
        "character_voice_context"
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
    if "primary_bid" in payload:
        result["primary_bid"] = payload["primary_bid"]
    if "semantic_relationship" in payload:
        result["semantic_relationship"] = payload["semantic_relationship"]
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
