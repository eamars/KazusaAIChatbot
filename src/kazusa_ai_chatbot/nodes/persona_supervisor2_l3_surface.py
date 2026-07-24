"""V2 L3 surface connector after the cognition state commit."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.action_spec.results import (
    project_trace_action_result_v2,
)
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    TextSurfaceInputV2,
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
    validate_cognition_core_output,
    validate_text_surface_input,
)
from kazusa_ai_chatbot.cognition_core_v2.surface import (
    run_text_surface_planning,
    run_visual_surface_planning,
)
from kazusa_ai_chatbot.db.interaction_style_images import (
    build_interaction_style_context,
)
from kazusa_ai_chatbot.llm_interface import LLMCallConfig
from kazusa_ai_chatbot.nodes.linguistic_texture import (
    get_abstraction_reframing_description,
    get_counter_questioning_description,
    get_direct_assertion_description,
    get_emotional_leakage_description,
    get_formalism_avoidance_description,
    get_fragmentation_description,
    get_hesitation_density_description,
    get_rhythmic_bounce_description,
    get_self_deprecation_description,
    get_softener_density_description,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import (
    build_runtime_capability_limits,
    _cognition_llm_config,
    _llm_interface,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState


_LINGUISTIC_TEXTURE_DESCRIPTORS = {
    "fragmentation": get_fragmentation_description,
    "hesitation_density": get_hesitation_density_description,
    "counter_questioning": get_counter_questioning_description,
    "softener_density": get_softener_density_description,
    "formalism_avoidance": get_formalism_avoidance_description,
    "abstraction_reframing": get_abstraction_reframing_description,
    "direct_assertion": get_direct_assertion_description,
    "emotional_leakage": get_emotional_leakage_description,
    "rhythmic_bounce": get_rhythmic_bounce_description,
    "self_deprecation": get_self_deprecation_description,
}


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
        "goal_resolution": validated_output["goal_resolution"],
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
        "character_voice_context": _character_voice_context(state),
    }
    runtime_limits = build_runtime_capability_limits(state)
    if runtime_limits:
        payload["runtime_capability_limits"] = runtime_limits
    admitted = validated_output.get("admitted_bid")
    if isinstance(admitted, Mapping):
        payload["primary_bid"] = _surface_bid_projection(admitted)
    relationship = validated_output.get("relationship_projection")
    if isinstance(relationship, Mapping):
        payload["semantic_relationship"] = dict(relationship)
    return validate_text_surface_input(payload)


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict[str, Any]:
    """Run sibling V2 text and enabled terminal visual surface planning."""

    interaction_style_context = await _load_interaction_style_context(state)
    input_payload = build_text_surface_input_from_global_state(
        state,
        interaction_style_context=interaction_style_context,
    )
    text_call = run_text_surface_planning(
        input_payload,
        _build_text_surface_services(),
    )
    if _visual_directives_disabled(input_payload):
        text_output = await text_call
        return_value = {"text_surface_output_v2": text_output}
        return return_value
    text_output, visual_output = await asyncio.gather(
        text_call,
        run_visual_surface_planning(
            input_payload,
            _build_visual_surface_services(),
        ),
    )
    return_value = {
        "text_surface_output_v2": text_output,
        "visual_surface_output_v2": visual_output,
    }
    return return_value


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
        "user_style": "当前用户风格",
        "group_channel_style": "当前群聊风格",
    }
    field_labels = {
        "speech_guidelines": "语言",
        "social_guidelines": "社交",
        "pacing_guidelines": "节奏",
        "engagement_guidelines": "互动",
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
        return "没有可用的已学习互动风格指引。"
    return " | ".join(fragments)


def _character_voice_context(state: Mapping[str, Any]) -> str:
    """Project the active profile into one bounded wording-only context."""

    profile = state.get("character_profile")
    if not isinstance(profile, Mapping):
        raise ValueError("character profile is required for surface planning")
    personality = profile["personality_brief"]
    linguistic_texture = profile["linguistic_texture_profile"]
    if not isinstance(personality, Mapping):
        raise ValueError("character personality brief must be a mapping")
    if not isinstance(linguistic_texture, Mapping):
        raise ValueError("character linguistic texture must be a mapping")
    field_labels = {
        "name": "姓名",
        "logic": "逻辑",
        "tempo": "节奏",
        "defense": "防御",
        "quirks": "特征",
        "taboos": "禁忌",
    }
    fragments = [
        f"{field_labels['name']}：{_voice_value(profile['name'], 80)}"
    ]
    for field_name in ("logic", "tempo", "defense", "quirks", "taboos"):
        fragments.append(
            f"{field_labels[field_name]}："
            f"{_voice_value(personality[field_name], 180)}"
        )
    texture_labels = {
        "fragmentation": "碎片化",
        "hesitation_density": "犹豫密度",
        "counter_questioning": "反问倾向",
        "softener_density": "缓和语密度",
        "formalism_avoidance": "正式化回避",
        "abstraction_reframing": "抽象重述",
        "direct_assertion": "直接断言",
        "emotional_leakage": "情绪泄露",
        "rhythmic_bounce": "节奏回弹",
        "self_deprecation": "自嘲",
    }
    for field_name, descriptor in _LINGUISTIC_TEXTURE_DESCRIPTORS.items():
        score = linguistic_texture[field_name]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("character linguistic texture score must be numeric")
        fragments.append(
            f"{texture_labels[field_name]}：{descriptor(float(score))}"
        )
    context = " | ".join(fragments)[:1500]
    return context


def _voice_value(value: object, maximum: int) -> str:
    """Render one required profile value into a bounded semantic fragment."""

    text = str(value).strip()
    if not text:
        raise ValueError("character voice value must be non-empty")
    return text[:maximum]


def _joined_length(fragments: list[str], candidate: str) -> int:
    """Return the rendered size after appending one candidate fragment."""

    separator_size = 3 if fragments else 0
    return len(" | ".join(fragments)) + separator_size + len(candidate)


def _build_text_surface_services() -> TextSurfaceServicesV2:
    """Bind the three V2 text-surface stages to the project LLM interface."""

    return TextSurfaceServicesV2(
        llm=_llm_interface,
        style_config=_surface_config("v2_surface_style"),
        content_plan_config=_surface_config("v2_surface_content"),
        preference_config=_surface_config("v2_surface_preference"),
    )


def _build_visual_surface_services() -> VisualSurfaceServicesV2:
    """Bind the terminal V2 visual stage to the project LLM interface."""

    return VisualSurfaceServicesV2(
        llm=_llm_interface,
        visual_config=_surface_config("v2_surface_visual"),
    )


def _visual_directives_disabled(payload: TextSurfaceInputV2) -> bool:
    """Return whether the canonical episode disables visual directives."""

    debug_modes = payload["episode"]["origin_metadata"]["debug_modes"]
    disabled = debug_modes.get("no_visual_directives") is True
    return disabled


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
        "motive": bid.get("reason", "有依据的分支"),
        "intention": bid["intention"],
        "desired_outcome": bid["desired_outcome"],
        "permitted_detail": bid["concrete_detail"],
        "target_summaries": [
            role.get("role", "对象")
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

    rows = state.get("pre_surface_action_results")
    if not isinstance(rows, list):
        return []
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        result.append(project_trace_action_result_v2(row))
    return result
