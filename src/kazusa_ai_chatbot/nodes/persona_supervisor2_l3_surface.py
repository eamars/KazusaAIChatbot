"""Canonical L3 surface connector after the cognition state commit."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot import llm_tracing
from kazusa_ai_chatbot.action_spec.results import (
    project_trace_action_result_v2,
)
from kazusa_ai_chatbot.background_work.result_source import (
    ToolResultCognitionSourceV1,
    validate_tool_result_cognition_source,
)
from kazusa_ai_chatbot.character_identity_growth.models import (
    TOP_LEVEL_IDENTITY_KEYS,
)
from kazusa_ai_chatbot.character_identity_growth.projection import (
    project_identity_for_surface,
)
from kazusa_ai_chatbot.cognition_resolver.contracts import (
    MAX_RESOLVER_EVIDENCE_EXCERPT_CHARS,
    MAX_RESOLVER_EVIDENCE_EXCERPTS,
    MAX_RESOLVER_GOAL_ITEM_CHARS,
    MAX_RESOLVER_GOAL_ITEMS,
    TERMINAL_RESOLVER_SURFACE_DECISION,
    resolver_evidence_excerpts_for_cognition,
)
from kazusa_ai_chatbot.cognition_resolver.state import validate_resolver_state
from kazusa_ai_chatbot.cognition_shared.contracts import (
    MAX_RECENT_CHARACTER_DIALOG_CHARS,
    MAX_RECENT_CHARACTER_DIALOG_ROWS,
    CognitionExecutionError,
    SurfaceAddresseePlanV1,
    TextSurfaceInput,
    TextSurfaceOutputV2,
    TextSurfaceServicesV2,
    VisualSurfaceServicesV2,
    validate_text_surface_input_canonical,
)
from kazusa_ai_chatbot.cognition_shared.model_attempt_policy import (
    V2_MODEL_TOTAL_ATTEMPTS,
)
from kazusa_ai_chatbot.cognition_shared.surface import (
    run_text_surface_planning,
    run_visual_surface_planning,
)
from kazusa_ai_chatbot.config import (
    COGNITION_STAGE_TIMEOUT_SECONDS,
    SURFACE_CONTENT_DEFAULT_MAX_COMPLETION_TOKENS,
    SURFACE_VISUAL_DEFAULT_MAX_COMPLETION_TOKENS,
)
from kazusa_ai_chatbot.conversation_progress import (
    project_conversation_progress_overused_moves,
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
    _cognition_llm_config,
    _llm_interface,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.runtime_coordination import PipelineCancelled
from kazusa_ai_chatbot.utils import build_interaction_history_recent

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
) -> TextSurfaceInput:
    """Build the exact surface contract from committed cognition output."""

    output = state.get("cognition_core_output")
    if not isinstance(output, Mapping):
        raise TypeError("canonical cognition output is required before surface planning")
    plan = output.get("response_plan")
    goal = output.get("active_character_goal")
    if not isinstance(plan, Mapping) or not isinstance(goal, Mapping):
        raise ValueError("canonical cognition response product is incomplete")
    private_monologue = output.get("private_monologue")
    epistemic_boundary = plan.get("epistemic_boundary")
    if not isinstance(private_monologue, str) or not private_monologue.strip():
        raise ValueError("canonical private monologue is required")
    if not isinstance(epistemic_boundary, str) or not epistemic_boundary.strip():
        raise ValueError("canonical epistemic boundary is required")
    continuation_ref = _resolver_result_continuation_ref(state)
    resolver_result = _resolver_result(
        state,
        continuation_ref=continuation_ref,
    )
    surface_response_plan = {
        key: plan[key]
        for key in (
            "response_goal",
            "goal_resolution",
            "epistemic_boundary",
            "action_requests",
            "resolver_requests",
        )
        if key in plan
    }
    terminal_surface = _terminal_resolver_surface_requirements(
        state,
        resolver_result=resolver_result,
    )
    if terminal_surface is not None:
        terminal_detail = terminal_surface["detail"]
        surface_response_plan.update({
            "response_goal": terminal_detail,
            "goal_resolution": "blocked",
            "action_requests": [],
            "resolver_requests": [],
            "surface_requirements": terminal_surface,
            "terminal_work_disposition": "closed",
        })
        epistemic_boundary = terminal_detail
    expression_context, visual_context = _character_surface_contexts(state)
    conversation_progress = state.get("conversation_progress")
    if conversation_progress is None:
        overused_moves: list[str] = []
    elif not isinstance(conversation_progress, dict):
        raise ValueError("conversation progress must be a canonical prompt mapping")
    else:
        overused_moves = project_conversation_progress_overused_moves(
            conversation_progress,
        )
    canonical_payload: TextSurfaceInput = {
        "schema_version": "text_surface_input.v4",
        "episode": _canonical_episode(state),
        "active_character_goal": dict(goal),
        "response_plan": surface_response_plan,
        "expression_policy": {
            "visibility": (
                "visible"
                if surface_response_plan.get("response_goal")
                else "none"
            ),
            "emotional_tone": "character-consistent",
            "intensity": "moderate",
            "directness": "balanced",
        },
        "semantic_affect": [dict(row) for row in output.get("affect_projection", [])],
        "overused_moves": overused_moves,
        "permitted_action_results": _action_results(state),
        "interaction_style_context": interaction_style_context,
        "character_expression_context": expression_context,
        "subjective_expression_context": {
            "private_monologue": private_monologue,
            "epistemic_boundary": epistemic_boundary,
        },
        "addressee_plan": _surface_addressee_plan(
            _surface_target_roles(state),
            state=state,
        ),
        "visual_character_context": visual_context,
        "recent_character_dialog": _recent_character_dialog(state),
    }
    willingness = output.get("relational_willingness")
    if isinstance(willingness, Mapping):
        canonical_payload["relational_willingness"] = dict(willingness)
    if resolver_result is not None:
        canonical_payload["resolver_result"] = resolver_result
    return validate_text_surface_input_canonical(canonical_payload)


def _terminal_resolver_surface_requirements(
    state: Mapping[str, Any],
    *,
    resolver_result: Mapping[str, Any] | None,
) -> dict[str, str] | None:
    """Return the resolver-owned closed-work surface for one terminal blocker."""

    if resolver_result is None or resolver_result.get("status") not in {
        "blocked",
        "failed",
    }:
        return None
    raw_resolver_state = state.get("resolver_state")
    if not isinstance(raw_resolver_state, Mapping):
        raise ValueError("terminal resolver surface requires resolver state")
    resolver_state = validate_resolver_state(raw_resolver_state)
    if resolver_state["status"] not in {"blocked", "max_cycles"}:
        return None

    action_specs = state.get("action_specs")
    if not isinstance(action_specs, list):
        raise TypeError("terminal resolver surface requires action specs")
    resolver_speak_specs = [
        row
        for row in action_specs
        if isinstance(row, Mapping)
        and row.get("kind") == "speak"
        and any(
            isinstance(source_ref, Mapping)
            and source_ref.get("owner") == "cognition_resolver"
            for source_ref in row.get("source_refs", [])
        )
    ]
    if len(resolver_speak_specs) != 1:
        raise ValueError(
            "terminal resolver surface requires exactly one resolver-owned speak"
        )
    params = resolver_speak_specs[0].get("params")
    if not isinstance(params, Mapping):
        raise ValueError("terminal resolver speak params are required")
    requirements = params.get("surface_requirements")
    if not isinstance(requirements, Mapping):
        raise ValueError("terminal resolver surface requirements are required")
    if set(requirements) != {"decision", "detail"}:
        raise ValueError("terminal resolver surface requirements are not exact")
    decision = requirements.get("decision")
    detail = requirements.get("detail")
    if decision != TERMINAL_RESOLVER_SURFACE_DECISION:
        raise ValueError("terminal resolver surface decision is invalid")
    if not isinstance(detail, str) or not detail.strip() or len(detail) > 1000:
        raise ValueError("terminal resolver surface detail is invalid")
    return {
        "decision": TERMINAL_RESOLVER_SURFACE_DECISION,
        "detail": detail.strip(),
    }


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict[str, Any]:
    """Run surface planning under the state-owned protected trace."""

    trace_token = llm_tracing.bind_trace_id(
        str(state.get("llm_trace_id") or "")
    )
    try:
        return await _run_l3_text_surface_handler(state)
    finally:
        llm_tracing.reset_trace_id(trace_token)


async def _run_l3_text_surface_handler(
    state: GlobalPersonaState,
) -> dict[str, Any]:
    """Run text and enabled terminal visual surface planning."""

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
        _assert_relational_willingness_preserved(input_payload, text_output)
        return_value = {
            "text_surface_input": input_payload,
            "text_surface_output_v2": text_output,
        }
        return return_value
    text_result, visual_result = await asyncio.gather(
        text_call,
        run_visual_surface_planning(
            input_payload,
            _build_visual_surface_services(),
        ),
        return_exceptions=True,
    )
    if isinstance(text_result, BaseException):
        raise text_result
    text_output = text_result
    _assert_relational_willingness_preserved(input_payload, text_output)
    return_value = {
        "text_surface_input": input_payload,
        "text_surface_output_v2": text_output,
    }
    if isinstance(visual_result, BaseException):
        if isinstance(visual_result, (asyncio.CancelledError, PipelineCancelled)):
            raise visual_result
        return_value["attempt_diagnostics"] = [{
            "schema_version": "episode_attempt_diagnostic.v1",
            "stage": "surface.visual",
            "error_code": "surface_visual_omitted",
            "attempt_count": V2_MODEL_TOTAL_ATTEMPTS,
            "safe_checkpoint": "post_cognition_commit",
            "retryable": False,
            "final_status": "accepted_degraded",
        }]
        return return_value
    return_value["visual_surface_output_v2"] = visual_result
    return return_value


def _assert_relational_willingness_preserved(
    surface_input: TextSurfaceInput,
    surface_output: TextSurfaceOutputV2,
) -> None:
    """Require surface planning to preserve the upstream typed stance exactly."""

    if surface_output.get("relational_willingness") != surface_input.get(
        "relational_willingness"
    ):
        raise CognitionExecutionError(
            "text surface changed the upstream relational willingness decision",
            error_code="surface_relational_willingness_mismatch",
            stage="surface.text",
            attempt_count=1,
            safe_checkpoint="pre_state_commit",
            retryable=False,
        )


async def _load_interaction_style_context(
    state: Mapping[str, Any],
) -> str:
    """Render the service-owned prompt-safe turn snapshot for L3."""

    context = state.get("interaction_style_context")
    if not isinstance(context, Mapping):
        raise ValueError("interaction style turn snapshot is required")
    return _render_interaction_style_context(context)


def _recent_character_dialog(state: Mapping[str, Any]) -> list[str]:
    """Project the latest visible messages authored by the current character."""

    history = state.get("chat_history_recent")
    if not isinstance(history, list):
        return []
    interaction_history = build_interaction_history_recent(
        history,
        str(state.get("platform_user_id", "") or ""),
        str(state.get("platform_bot_id", "") or ""),
        str(state.get("global_user_id", "") or ""),
    )
    projected: list[str] = []
    for row in interaction_history:
        if row.get("role") != "assistant":
            continue
        body_text = row.get("body_text")
        if not isinstance(body_text, str):
            body_text = row.get("content")
        if not isinstance(body_text, str):
            continue
        body_text = body_text.strip()
        if not body_text:
            continue
        projected.append(body_text[:MAX_RECENT_CHARACTER_DIALOG_CHARS])
    return projected[-MAX_RECENT_CHARACTER_DIALOG_ROWS:]


def _render_interaction_style_context(context: Mapping[str, Any]) -> str:
    """Project allowlisted style guidance into the bounded text field."""

    if context.get("schema_version") != "interaction_style_turn_snapshot.v1":
        raise ValueError("interaction style snapshot schema is invalid")
    application_order = context.get("application_order")
    if not isinstance(application_order, list):
        raise ValueError("interaction style application order is required")
    surface = context.get("surface")
    if not isinstance(surface, Mapping):
        raise ValueError("interaction style surface projection is required")

    scope_labels = {
        "user": "当前用户风格",
        "group_channel": "当前群聊风格",
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
        source_projection = surface.get(scope_name)
        if not isinstance(source_projection, Mapping):
            raise ValueError("interaction style source projection is required")
        overlay = source_projection.get("overlay")
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


def _character_surface_contexts(
    state: Mapping[str, Any],
) -> tuple[dict[str, str], str]:
    """Project delivery-only text context and isolated visual context.

    Args:
        state: Persona state containing the validated character profile.

    Returns:
        The bounded text-expression context and visual-only profile context.
    """

    projected = state.get("character_identity_surface_context")
    if not isinstance(projected, Mapping):
        profile = state.get("character_profile")
        if not isinstance(profile, Mapping):
            raise ValueError(
                "character profile is required for surface planning"
            )
        effective_identity = {
            key: profile[key]
            for key in TOP_LEVEL_IDENTITY_KEYS
        }
        projected = project_identity_for_surface({
            "effective_identity": effective_identity,
        })
    if set(projected) != {"text", "visual", "naming"}:
        raise ValueError(
            "character surface identity must contain exact consumers"
        )
    text_identity = projected["text"]
    visual_identity = projected["visual"]
    if not isinstance(text_identity, Mapping):
        raise ValueError("character text identity must be a mapping")
    if not isinstance(visual_identity, Mapping):
        raise ValueError("character visual identity must be a mapping")
    personality = text_identity["personality"]
    linguistic_texture = text_identity["linguistic_texture_profile"]
    if not isinstance(personality, Mapping):
        raise ValueError("character personality brief must be a mapping")
    if not isinstance(linguistic_texture, Mapping):
        raise ValueError("character linguistic texture must be a mapping")
    expression_fragments = [
        f"姓名：{_profile_text(text_identity['name'], 80)}",
        f"防御：{_profile_text(personality['defense'], 180)}",
        f"特征：{_profile_text(personality['quirks'], 180)}",
    ]
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
    texture_fragments: list[str] = []
    for field_name, descriptor in _LINGUISTIC_TEXTURE_DESCRIPTORS.items():
        score = linguistic_texture[field_name]
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise ValueError("character linguistic texture score must be numeric")
        texture_fragments.append(
            f"{texture_labels[field_name]}：{descriptor(float(score))}"
        )
    texture_context = " | ".join(texture_fragments)[:1000]
    expression_context = {
        "tempo": _profile_text(personality["tempo"], 180),
        "linguistic_texture": " | ".join([
            *expression_fragments,
            texture_context,
        ])[:1000],
    }
    visual_fragments = [
        f"姓名：{_profile_text(visual_identity['name'], 80)}",
        f"描述：{_profile_text(visual_identity['description'], 500)}",
        f"性别：{_profile_text(visual_identity['gender'], 80)}",
        f"年龄：{visual_identity['age']}",
        "视觉特征："
        f"{_profile_text(visual_identity['visual_characterization'], 700)}",
    ]
    visual_context = " | ".join(visual_fragments)[:1500]
    return expression_context, visual_context


def _profile_text(value: object, maximum: int) -> str:
    """Render one required profile value into a bounded semantic fragment."""

    if not isinstance(value, str):
        raise ValueError("character profile value must be text")
    text = value.strip()
    if not text:
        raise ValueError("character profile value must be non-empty")
    return text[:maximum]


def _joined_length(fragments: list[str], candidate: str) -> int:
    """Return the rendered size after appending one candidate fragment."""

    separator_size = 3 if fragments else 0
    return len(" | ".join(fragments)) + separator_size + len(candidate)


def _build_text_surface_services() -> TextSurfaceServicesV2:
    """Bind the text-surface stages to the project LLM interface."""

    return TextSurfaceServicesV2(
        llm=_llm_interface,
        content_plan_config=_surface_config(
            "v2_surface_content",
            max_completion_tokens=(
                SURFACE_CONTENT_DEFAULT_MAX_COMPLETION_TOKENS
            ),
        ),
    )


def _build_visual_surface_services() -> VisualSurfaceServicesV2:
    """Bind the terminal visual stage to the project LLM interface."""

    return VisualSurfaceServicesV2(
        llm=_llm_interface,
        visual_config=_surface_config(
            "v2_surface_visual",
            max_completion_tokens=(
                SURFACE_VISUAL_DEFAULT_MAX_COMPLETION_TOKENS
            ),
        ),
    )


def _visual_directives_disabled(payload: TextSurfaceInput) -> bool:
    """Return whether the canonical episode disables visual directives."""

    debug_modes = payload["episode"]["origin_metadata"]["debug_modes"]
    disabled = debug_modes.get("no_visual_directives") is True
    return disabled


def _surface_config(
    stage_name: str,
    *,
    max_completion_tokens: int,
) -> LLMCallConfig:
    """Bind one surface stage to the cognition route with its own budget."""

    return LLMCallConfig(
        stage_name=stage_name,
        route_name=_cognition_llm_config.route_name,
        base_url=_cognition_llm_config.base_url,
        api_key=_cognition_llm_config.api_key,
        model=_cognition_llm_config.model,
        temperature=_cognition_llm_config.temperature,
        top_p=_cognition_llm_config.top_p,
        top_k=_cognition_llm_config.top_k,
        max_completion_tokens=max_completion_tokens,
        presence_penalty=_cognition_llm_config.presence_penalty,
        timeout_seconds=COGNITION_STAGE_TIMEOUT_SECONDS,
        thinking=_cognition_llm_config.thinking,
    )


def _surface_bid_projection(
    bid: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    """Copy complete-bid semantic content without persistent ids or private refs."""

    return {
        "motive": bid.get("reason", "有依据的分支"),
        "intention": bid["intention"],
        "desired_outcome": bid["desired_outcome"],
        "permitted_detail": bid["concrete_detail"],
        "target_summaries": [
            _surface_role_summary(role, state=state)
            for role in bid.get("target_roles", [])
            if isinstance(role, Mapping)
        ],
        "expected_consequences": list(bid["expected_consequences"]),
    }


def _surface_role_summary(
    role: Mapping[str, Any],
    *,
    state: Mapping[str, Any],
) -> str:
    """Render a target summary with visible names but no backing IDs."""

    entity_kind = role.get("entity_kind")
    if entity_kind == "third_party":
        handle = _role_episode_handle(role)
        for binding in state.get("scene_participant_bindings", []):
            if not isinstance(binding, Mapping):
                continue
            if binding.get("handle") != handle:
                continue
            display_name = binding.get("display_name")
            if isinstance(display_name, str) and display_name.strip():
                return display_name.strip()
        return "群聊其他参与者"
    if entity_kind == "user":
        return str(state.get("user_name", "当前用户")) or "当前用户"
    if entity_kind == "character":
        profile = state.get("character_profile")
        if isinstance(profile, Mapping):
            name = profile.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        return "当前角色"
    return str(role.get("role", "对象"))


def _role_episode_handle(role: Mapping[str, Any]) -> str:
    """Extract the non-persistent handle from a third-party role reference."""

    entity_id = role.get("entity_id")
    if not isinstance(entity_id, str) or not entity_id.startswith("scene:"):
        return ""
    return entity_id.removeprefix("scene:")


def _surface_addressee_plan(
    target_roles: object,
    *,
    state: Mapping[str, Any],
) -> list[SurfaceAddresseePlanV1]:
    """Project admitted target roles into visible wording constraints."""

    if not isinstance(target_roles, list):
        raise ValueError("surface target roles must be a list")
    result: list[SurfaceAddresseePlanV1] = []
    seen_handles: set[str] = set()
    for role in target_roles:
        if not isinstance(role, Mapping):
            raise ValueError("surface target role must be a mapping")
        entity_kind = role.get("entity_kind")
        if entity_kind == "third_party":
            handle = _role_episode_handle(role)
            binding = next(
                (
                    row
                    for row in state.get("scene_participant_bindings", [])
                    if isinstance(row, Mapping)
                    and row.get("handle") == handle
                ),
                None,
            )
            if not isinstance(binding, Mapping):
                raise ValueError("surface target third-party binding is missing")
            display_name = binding.get("display_name")
            if not isinstance(display_name, str) or not display_name.strip():
                raise ValueError("surface target display name is invalid")
            semantic_role = "embedded_target"
            wording_policy = "named_or_third_person_required"
        elif entity_kind == "user":
            handle = "current_user"
            display_name = str(state.get("user_name", "当前用户"))
            semantic_role = "direct_recipient"
            wording_policy = "second_person_allowed"
        elif entity_kind == "character":
            handle = "self"
            profile = state.get("character_profile")
            display_name = (
                str(profile.get("name", "当前角色"))
                if isinstance(profile, Mapping)
                else "当前角色"
            )
            semantic_role = "embedded_actor"
            wording_policy = "named_or_third_person_required"
        else:
            continue
        if handle in seen_handles:
            continue
        seen_handles.add(handle)
        result.append({
            "handle": handle,
            "display_name": display_name.strip(),
            "semantic_role": semantic_role,
            "wording_policy": wording_policy,
        })
    return result


def _surface_target_roles(
    state: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Read the caller-owned target roles from the selected speak action."""

    action_specs = state.get("action_specs")
    if not isinstance(action_specs, list):
        return []
    speak_specs = [
        row
        for row in action_specs
        if isinstance(row, Mapping) and row.get("kind") == "speak"
    ]
    if len(speak_specs) > 1:
        raise ValueError("surface received multiple speak action specs")
    if not speak_specs:
        return []
    provenance = speak_specs[0].get("cognition_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("speak action cognition provenance is required")
    target_roles = provenance.get("target_roles")
    if not isinstance(target_roles, list):
        raise ValueError("speak action target roles must be a list")
    if any(not isinstance(role, Mapping) for role in target_roles):
        raise ValueError("speak action target role must be a mapping")
    return [dict(role) for role in target_roles]


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


def _resolver_result_continuation_ref(
    state: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Resolve the exact typed continuation reference for one L3 result."""

    episode = state.get("cognitive_episode")
    if isinstance(episode, Mapping) and episode.get("trigger_source") == "tool_result":
        return _tool_result_surface_continuation_ref(state, episode)
    raw_resolver_state = state.get("resolver_state")
    if not isinstance(raw_resolver_state, Mapping):
        return None
    resolver_state = validate_resolver_state(raw_resolver_state)
    dependency = resolver_state.get("required_resolver_evidence_dependency")
    if dependency is not None:
        return dependency["goal_continuation_ref"]
    observations = resolver_state["observations"]
    if (
        not observations
        or observations[-1]["capability_kind"] != "task_resolution_request"
    ):
        return None
    continuation_ref = observations[-1].get("goal_continuation_ref")
    if not isinstance(continuation_ref, Mapping):
        raise TypeError("task resolver observation lacks a continuation reference")
    return continuation_ref


def _tool_result_surface_continuation_ref(
    state: Mapping[str, Any],
    episode: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Cross-check one tool-result source against its cognition-owned speak."""

    typed_source = _episode_tool_result_source(episode)
    continuation_ref = typed_source["goal_continuation_ref"]
    origin = episode.get("origin_metadata")
    if (
        not isinstance(origin, Mapping)
        or origin.get("goal_continuation_ref") != continuation_ref
    ):
        raise ValueError("tool-result episode origin continuation reference conflicts")
    action_specs = state.get("action_specs")
    if not isinstance(action_specs, list):
        raise TypeError("tool-result surface requires cognition-owned action specs")
    speak_specs = [
        row
        for row in action_specs
        if isinstance(row, Mapping) and row.get("kind") == "speak"
    ]
    if len(speak_specs) != 1:
        raise ValueError("tool-result surface requires exactly one speak action")
    expected_role = (
        "task_result"
        if typed_source["task_status"] in {"resolved", "partial"}
        else "task_status"
    )
    speak_spec = speak_specs[0]
    if speak_spec.get("surface_role") != expected_role:
        raise ValueError("tool-result speak surface role conflicts with result status")
    if speak_spec.get("goal_continuation_ref") != continuation_ref:
        raise ValueError("tool-result speak continuation reference conflicts")
    return continuation_ref


def _resolver_result(
    state: Mapping[str, Any],
    *,
    continuation_ref: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """Project the bound source-owned resolver outcome into L3."""

    episode = state.get("cognitive_episode")
    if (
        isinstance(episode, Mapping)
        and episode.get("trigger_source") == "tool_result"
    ):
        return _tool_result_resolver_result(episode, continuation_ref)

    raw_resolver_state = state.get("resolver_state")
    if not isinstance(raw_resolver_state, Mapping):
        return None
    resolver_state = validate_resolver_state(raw_resolver_state)
    observations = resolver_state["observations"]
    if not observations:
        return None
    dependency = resolver_state.get(
        "required_resolver_evidence_dependency"
    )
    if dependency is not None:
        observation = next(
            (
                candidate
                for candidate in observations
                if candidate["observation_id"] == dependency["observation_id"]
            ),
            None,
        )
        if observation is None:
            raise ValueError(
                "required resolver evidence observation is unavailable"
            )
        if observation["capability_kind"] != "task_resolution_request":
            raise ValueError(
                "required resolver evidence observation has wrong capability"
            )
        return _task_resolver_result(
            observation,
            dependency=dependency,
            continuation_ref=continuation_ref,
        )

    observation = observations[-1]
    if observation["capability_kind"] == "task_resolution_request":
        return _task_resolver_result(
            observation,
            dependency=None,
            continuation_ref=continuation_ref,
        )
    return {
        "capability_kind": observation["capability_kind"],
        "status": observation["status"],
        "semantic_result": observation["prompt_safe_summary"],
    }


_TOOL_RESULT_SURFACE_STATUS = {
    "resolved": "succeeded",
    "partial": "succeeded",
    "deferred": "succeeded",
    "needs_user_input": "blocked",
    "approval_required": "blocked",
    "unavailable": "failed",
    "failed": "failed",
}
_TOOL_RESULT_OBSERVATION_HANDLE = "tool_result_episode"


def _tool_result_resolver_result(
    episode: Mapping[str, Any],
    continuation_ref: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project the stored typed task result as the L3 source-owned authority.

    A later tool-result episode carries its own typed result contract in the
    percept; current-cycle resolver observations never replace it. Factual
    states expose only validated evidence excerpts, and every non-factual
    state keeps empty evidence so surface wording stays objective-scoped
    status or clarification.
    """

    origin = episode.get("origin_metadata")
    if not isinstance(origin, Mapping):
        raise ValueError("tool-result episode origin is required")
    if origin.get("goal_continuation_ref") != continuation_ref:
        raise ValueError(
            "tool-result episode origin continuation reference conflicts "
            "with the committed cognition reference"
        )
    typed_source = _episode_tool_result_source(episode)
    if typed_source["goal_continuation_ref"] != continuation_ref:
        raise ValueError(
            "tool-result cognition source continuation reference conflicts "
            "with the committed cognition reference"
        )
    surface_status = _TOOL_RESULT_SURFACE_STATUS[typed_source["task_status"]]
    return {
        "capability_kind": "task_resolution_request",
        "status": surface_status,
        "semantic_result": typed_source["semantic_summary"],
        "prompt_safe_observation_handle": _TOOL_RESULT_OBSERVATION_HANDLE,
        "evidence_state": typed_source["evidence_state"],
        "evidence_excerpts": [
            excerpt[:MAX_RESOLVER_EVIDENCE_EXCERPT_CHARS]
            for excerpt in typed_source["evidence_excerpts"][
                :MAX_RESOLVER_EVIDENCE_EXCERPTS
            ]
        ],
        "evidence_handles": [
            handle[:MAX_RESOLVER_GOAL_ITEM_CHARS]
            for handle in typed_source["evidence_handles"][
                :MAX_RESOLVER_EVIDENCE_EXCERPTS
            ]
        ],
        "remaining_needs": [
            need[:MAX_RESOLVER_GOAL_ITEM_CHARS]
            for need in typed_source["remaining_needs"][
                :MAX_RESOLVER_GOAL_ITEMS
            ]
        ],
    }


def _episode_tool_result_source(
    episode: Mapping[str, Any],
) -> ToolResultCognitionSourceV1:
    """Require the validated typed tool-result source on one result percept."""

    percepts = episode.get("percepts")
    if not isinstance(percepts, list):
        raise ValueError("tool-result episode percepts are required")
    for percept in percepts:
        if not isinstance(percept, Mapping):
            continue
        if percept.get("source_kind") != "tool_result":
            continue
        content = percept.get("content")
        if not isinstance(content, Mapping):
            raise ValueError("tool-result percept content is required")
        raw_source = content.get("cognition_source")
        if not isinstance(raw_source, Mapping):
            raise ValueError(
                "tool-result percept requires a typed cognition_source"
            )
        try:
            typed_source = validate_tool_result_cognition_source(raw_source)
        except ValueError as exc:
            raise ValueError(
                f"tool-result cognition_source is invalid: {exc}"
            ) from exc
        return typed_source
    raise ValueError(
        "tool-result episode requires a typed tool-result percept"
    )


def _task_resolver_result(
    observation: Mapping[str, Any],
    *,
    dependency: Mapping[str, Any] | None,
    continuation_ref: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Project one task observation with source-owned evidence metadata.

    The observation and any required dependency must carry the exact
    goal-continuation reference committed by cognition; a missing or
    mismatched reference fails closed so no result wording can detach from
    its original goal.
    """

    observation_ref = observation.get("goal_continuation_ref")
    if observation_ref != continuation_ref:
        raise ValueError(
            "task resolver observation continuation reference conflicts with "
            "the committed cognition reference"
        )
    evidence_state = observation.get("task_resolution_evidence_state")
    if not isinstance(evidence_state, Mapping):
        raise ValueError("task resolver observation lacks evidence state")
    excerpts = resolver_evidence_excerpts_for_cognition(observation)
    if dependency is None:
        prompt_safe_observation_handle = "resolver_observation_optional"
        evidence_handles = [
            f"resolver_evidence_optional_{index}"
            for index, _excerpt in enumerate(excerpts, start=1)
        ]
    else:
        if dependency["goal_continuation_ref"] != continuation_ref:
            raise ValueError(
                "required resolver dependency continuation reference conflicts "
                "with the committed cognition reference"
            )
        prompt_safe_observation_handle = dependency[
            "prompt_safe_observation_handle"
        ]
        evidence_handles = list(dependency["evidence_handles"])
        if len(evidence_handles) != len(excerpts):
            raise ValueError(
                "required resolver evidence handles do not match excerpts"
            )
        if dependency["state"] != evidence_state["state"]:
            raise ValueError(
                "required resolver evidence state does not match observation"
            )
        if list(dependency["remaining_needs"]) != list(
            evidence_state["remaining_needs"]
        ):
            raise ValueError(
                "required resolver remaining needs do not match observation"
            )
    return {
        "capability_kind": observation["capability_kind"],
        "status": observation["status"],
        "semantic_result": observation["prompt_safe_summary"],
        "prompt_safe_observation_handle": prompt_safe_observation_handle,
        "evidence_state": evidence_state["state"],
        "evidence_excerpts": excerpts,
        "evidence_handles": evidence_handles,
        "remaining_needs": list(evidence_state["remaining_needs"]),
    }
