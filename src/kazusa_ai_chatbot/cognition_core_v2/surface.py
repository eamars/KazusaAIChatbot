"""Public V2 text-surface planning facade."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceInputV2,
    TextSurfaceOutputV2,
    TextSurfaceServicesV2,
    validate_text_surface_input,
    validate_text_surface_output,
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
    """Run four bounded surface stages after cognition state is committed."""

    payload = validate_text_surface_input(input_payload)
    stage_payload = _project_surface_payload(payload)
    validate_prompt_projection(stage_payload)
    stage_calls = [
        run_style_stage(stage_payload, services),
        run_content_plan_stage(stage_payload, services),
        run_preference_stage(stage_payload, services),
    ]
    if _visual_guidance_disabled(payload["episode"]):
        style, content, preference = await asyncio.gather(*stage_calls)
        visual = "Visual and pacing guidance is disabled for this episode."
    else:
        style, content, preference, visual = await asyncio.gather(
            *stage_calls,
            run_visual_stage(stage_payload, services),
        )
    visible_boundaries, addressee_plan = preference
    output: TextSurfaceOutputV2 = {
        "schema_version": "text_surface_output.v2",
        "content_plan": content,
        "visible_boundaries": visible_boundaries,
        "addressee_plan": addressee_plan,
        "style_guidance": style,
        "pacing_guidance": visual,
        "selected_surface_intent": payload["intention"]["intention"],
    }
    return validate_text_surface_output(output)


def _visual_guidance_disabled(episode: Mapping[str, Any]) -> bool:
    """Return the deterministic episode flag for skipping visual planning."""

    origin_metadata = episode.get("origin_metadata")
    if not isinstance(origin_metadata, Mapping):
        return False
    debug_modes = origin_metadata.get("debug_modes")
    return isinstance(debug_modes, Mapping) and (
        debug_modes.get("no_visual_directives") is True
    )


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
        "supporting_bids": payload["supporting_bids"],
        "expression_policy": payload["expression_policy"],
        "semantic_affect": payload["semantic_affect"],
        "permitted_action_results": payload["permitted_action_results"],
        "interaction_style_context": payload["interaction_style_context"],
    }
    if "primary_bid" in payload:
        result["primary_bid"] = payload["primary_bid"]
    if "semantic_relationship" in payload:
        result["semantic_relationship"] = payload["semantic_relationship"]
    return result


def _project_episode(episode: Mapping[str, Any]) -> dict[str, Any]:
    """Project visible typed percepts and configured-local time for L3."""

    visible_percepts = [
        {
            "input_source": percept["input_source"],
            "content": percept["content"],
        }
        for percept in episode["percepts"]
        if percept["visibility"] == "model_visible"
    ]
    local_time = episode["local_time_context"]
    return {
        "visible_percepts": visible_percepts,
        "local_time_context": {
            "current_local_datetime": local_time["current_local_datetime"],
            "current_local_weekday": local_time["current_local_weekday"],
        },
    }
