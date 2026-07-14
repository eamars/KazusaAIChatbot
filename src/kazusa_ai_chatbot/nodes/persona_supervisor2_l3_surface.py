"""V2 L3 surface connector after the cognition state commit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2 import run_text_surface_planning
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceInputV2,
    TextSurfaceServicesV2,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    _cognition_llm_config,
    _llm_interface,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.utils import parse_llm_json_output


def build_text_surface_input_from_global_state(
    state: GlobalPersonaState,
) -> TextSurfaceInputV2:
    """Build the exact surface contract from committed V2 cognition output."""

    output = state.get("cognition_core_output")
    if not isinstance(output, Mapping):
        raise ValueError("V2 cognition output is required before surface planning")
    payload: TextSurfaceInputV2 = {
        "schema_version": "text_surface_input.v2",
        "episode": _episode_projection(state),
        "intention": dict(output["intention"]),
        "supporting_bids": [
            _surface_bid_projection(bid)
            for bid in output.get("supporting_bids", [])
        ],
        "expression_policy": dict(output["expression_policy"]),
        "semantic_affect": [
            dict(row) for row in output.get("affect_projection", [])
        ],
        "permitted_action_results": _action_results(state),
        "interaction_style_context": "bounded character style context",
    }
    admitted = output.get("admitted_bid")
    if isinstance(admitted, Mapping):
        payload["primary_bid"] = _surface_bid_projection(admitted)
    relationship = output.get("relationship_projection")
    if isinstance(relationship, Mapping):
        payload["semantic_relationship"] = dict(relationship)
    return payload


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict[str, Any]:
    """Run V2 surface planning and hand the canonical output to dialog."""

    input_payload = build_text_surface_input_from_global_state(state)
    output = await run_text_surface_planning(
        input_payload,
        _build_surface_services(),
    )
    return {"text_surface_output_v2": output}


def _build_surface_services() -> TextSurfaceServicesV2:
    """Bind the four V2 surface stages to the project LLM interface."""

    return TextSurfaceServicesV2(
        llm=_llm_interface,
        style_config=_surface_config("v2_surface_style"),
        content_plan_config=_surface_config("v2_surface_content"),
        preference_config=_surface_config("v2_surface_preference"),
        visual_config=_surface_config("v2_surface_visual"),
        parse_json=parse_llm_json_output,
        logger=_NullLogger(),
    )


def _surface_config(stage_name: str) -> LLMCallConfig:
    """Reuse the configured cognition route with a stage-specific identity."""

    return LLMCallConfig(
        stage_name=stage_name,
        route_name=_cognition_llm_config.route_name,
        base_url=_cognition_llm_config.base_url,
        api_key=_cognition_llm_config.api_key,
        model=_cognition_llm_config.model,
        temperature=_cognition_llm_config.temperature,
        top_p=_cognition_llm_config.top_p,
        top_k=_cognition_llm_config.top_k,
        max_completion_tokens=_cognition_llm_config.max_completion_tokens,
        presence_penalty=_cognition_llm_config.presence_penalty,
        thinking=_cognition_llm_config.thinking,
    )


def _surface_bid_projection(bid: Mapping[str, Any]) -> dict[str, Any]:
    """Copy complete-bid semantic content without persistent ids or private refs."""

    return {
        "motive": bid.get("reason", "grounded branch"),
        "intention": bid["intention"],
        "desired_outcome": bid["desired_outcome"],
        "permitted_detail": bid["concrete_detail"],
        "target_summaries": [
            role.get("role", "target")
            for role in bid.get("target_roles", [])
            if isinstance(role, Mapping)
        ],
        "expected_consequences": list(bid["expected_consequences"]),
    }


def _episode_projection(state: Mapping[str, Any]) -> dict[str, str]:
    """Project semantic episode context for surface stages."""

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        episode = {}
    return {
        "episode_summary": str(episode.get("semantic_scene", state.get("user_input", ""))),
        "semantic_scene": str(episode.get("semantic_scene", "conversation")),
        "semantic_temporal_context": "immediate",
    }


def _action_results(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project already-permitted action results into the surface contract."""

    rows = state.get("action_results")
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        result.append({
            "action_kind": str(row.get("action_kind", "unknown")),
            "status": str(row.get("status", "unavailable")),
            "semantic_result": str(row.get("result_summary", "")),
            "target_roles": [],
        })
    return result


class _NullLogger:
    """No-op surface logger."""

    def debug(self, message: str, *args: object, **kwargs: object) -> None:
        del message, args, kwargs

    info = debug
    warning = debug
    error = debug
