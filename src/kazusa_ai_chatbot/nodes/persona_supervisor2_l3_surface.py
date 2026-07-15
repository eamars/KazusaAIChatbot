"""V2 L3 surface connector after the cognition state commit."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.action_spec.results import (
    project_trace_action_result_v2,
)
from kazusa_ai_chatbot.cognition_core_v2 import run_text_surface_planning
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceInputV2,
    TextSurfaceServicesV2,
    validate_cognition_core_output,
    validate_text_surface_input,
)
from kazusa_ai_chatbot.db.interaction_style_images import (
    build_interaction_style_context,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    _cognition_llm_config,
    _llm_interface,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState


def build_text_surface_input_from_global_state(
    state: GlobalPersonaState,
    *,
    interaction_style_context: str,
) -> TextSurfaceInputV2:
    """Build the exact surface contract from committed V2 cognition output."""

    output = state.get("cognition_core_output")
    if not isinstance(output, Mapping):
        raise ValueError("V2 cognition output is required before surface planning")
    validated_output = validate_cognition_core_output(output)
    payload: TextSurfaceInputV2 = {
        "schema_version": "text_surface_input.v2",
        "episode": _canonical_episode(state),
        "intention": dict(validated_output["intention"]),
        "supporting_bids": [
            _surface_bid_projection(bid)
            for bid in validated_output["supporting_bids"]
        ],
        "expression_policy": dict(validated_output["expression_policy"]),
        "semantic_affect": [
            dict(row) for row in validated_output["affect_projection"]
        ],
        "permitted_action_results": _action_results(state),
        "interaction_style_context": interaction_style_context,
    }
    admitted = validated_output.get("admitted_bid")
    if isinstance(admitted, Mapping):
        payload["primary_bid"] = _surface_bid_projection(admitted)
    relationship = validated_output.get("relationship_projection")
    if isinstance(relationship, Mapping):
        payload["semantic_relationship"] = dict(relationship)
    return validate_text_surface_input(payload)


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict[str, Any]:
    """Run V2 surface planning and hand the canonical output to dialog."""

    interaction_style_context = await _load_interaction_style_context(state)
    input_payload = build_text_surface_input_from_global_state(
        state,
        interaction_style_context=interaction_style_context,
    )
    output = await run_text_surface_planning(
        input_payload,
        _build_surface_services(),
    )
    return {"text_surface_output_v2": output}


async def _load_interaction_style_context(
    state: Mapping[str, Any],
) -> str:
    """Load and render prompt-safe style guidance for the active surface."""

    context = await build_interaction_style_context(
        global_user_id=str(state.get("global_user_id", "")),
        channel_type=str(state.get("channel_type", "")),
        platform=str(state.get("platform", "")),
        platform_channel_id=str(state.get("platform_channel_id", "")),
    )
    return _render_interaction_style_context(context)


def _render_interaction_style_context(context: Mapping[str, Any]) -> str:
    """Project allowlisted style guidance into the bounded V2 text field."""

    application_order = context.get("application_order")
    if not isinstance(application_order, list):
        raise ValueError("interaction style application order is required")

    scope_labels = {
        "user_style": "Current participant style",
        "group_channel_style": "Current group style",
    }
    field_labels = {
        "speech_guidelines": "speech",
        "social_guidelines": "social",
        "pacing_guidelines": "pacing",
        "engagement_guidelines": "engagement",
    }
    fragments: list[str] = []
    for scope_name in application_order:
        if scope_name not in scope_labels:
            raise ValueError("unknown interaction style scope")
        overlay = context.get(scope_name)
        if not isinstance(overlay, Mapping):
            raise ValueError("interaction style overlay is required")
        for field_name, field_label in field_labels.items():
            guidelines = overlay.get(field_name)
            if not isinstance(guidelines, list):
                raise ValueError("interaction style guidelines must be a list")
            for guideline in guidelines:
                if not isinstance(guideline, str) or not guideline.strip():
                    raise ValueError("interaction style guideline must be text")
                candidate = (
                    f"{scope_labels[scope_name]} {field_label}: "
                    f"{guideline.strip()}"
                )
                if _joined_length(fragments, candidate) <= 500:
                    fragments.append(candidate)

    if not fragments:
        return "No learned interaction style guidance is available."
    return " | ".join(fragments)


def _joined_length(fragments: list[str], candidate: str) -> int:
    """Return the rendered size after appending one candidate fragment."""

    separator_size = 3 if fragments else 0
    return len(" | ".join(fragments)) + separator_size + len(candidate)


def _build_surface_services() -> TextSurfaceServicesV2:
    """Bind the four V2 surface stages to the project LLM interface."""

    return TextSurfaceServicesV2(
        llm=_llm_interface,
        style_config=_surface_config("v2_surface_style"),
        content_plan_config=_surface_config("v2_surface_content"),
        preference_config=_surface_config("v2_surface_preference"),
        visual_config=_surface_config("v2_surface_visual"),
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


def _canonical_episode(state: Mapping[str, Any]) -> dict[str, Any]:
    """Pass the canonical episode to the validated public L3 boundary."""

    episode = state.get("cognitive_episode")
    if not isinstance(episode, dict):
        raise ValueError("canonical cognitive episode is required")
    return dict(episode)


def _action_results(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project already-permitted action results into the surface contract."""

    rows = state.get("action_results")
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        result.append(project_trace_action_result_v2(row))
    return result
