"""Selected text-surface planning entrypoint."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionChainServices,
    CognitionTextSurfaceInputV1,
    CognitionTextSurfaceOutputV1,
    LLMStageBinding,
    validate_text_surface_input,
    validate_text_surface_output,
)
from kazusa_ai_chatbot.cognition_chain_core.episode_projection import (
    build_prompt_selection_episode,
)
from kazusa_ai_chatbot.cognition_chain_core.graph_state import CoreStageState
from kazusa_ai_chatbot.cognition_chain_core.stages import l3
from kazusa_ai_chatbot.cognition_chain_core.utils import (
    empty_user_memory_context,
    reset_json_parser,
    set_json_parser,
)

ServiceBinding = tuple[Callable[[Any], None], Any]


async def run_text_surface_planning(
    input_payload: CognitionTextSurfaceInputV1,
    services: CognitionChainServices,
) -> CognitionTextSurfaceOutputV1:
    """Run selected text-surface planning from the public surface contract."""

    validated_input = validate_text_surface_input(input_payload)
    initial_state = _state_from_surface_input(validated_input)
    graph_result = await run_text_surface_state_graph(initial_state, services)
    output: CognitionTextSurfaceOutputV1 = {
        "schema_version": "cognition_text_surface_output.v1",
        "action_directives": graph_result["action_directives"],
    }
    validated_output = validate_text_surface_output(output)
    return validated_output


async def run_text_surface_state_graph(
    initial_state: Mapping[str, Any],
    services: CognitionChainServices,
) -> dict[str, Any]:
    """Run the moved selected-surface graph over projected stage state."""

    service_bindings = _inject_surface_services(services)
    try:
        surface_builder = StateGraph(CoreStageState)
        surface_builder.add_node(
            "l3_interaction_style_context_loader",
            l3.call_interaction_style_context_loader,
        )
        surface_builder.add_node("l3_style_agent", l3.call_style_agent)
        surface_builder.add_node(
            "l3_content_plan_agent",
            l3.call_content_plan_agent,
        )
        surface_builder.add_node(
            "l3_preference_adapter",
            l3.call_preference_adapter,
        )
        surface_builder.add_node("l3_visual_agent", l3.call_visual_agent)
        surface_builder.add_node(
            "l4_surface_directive_collector",
            l3.call_surface_directive_collector,
        )

        surface_builder.add_edge(START, "l3_interaction_style_context_loader")
        surface_builder.add_edge(
            "l3_interaction_style_context_loader",
            "l3_content_plan_agent",
        )
        surface_builder.add_edge("l3_content_plan_agent", "l3_visual_agent")
        surface_builder.add_edge(
            "l3_interaction_style_context_loader",
            "l3_style_agent",
        )
        surface_builder.add_edge(
            ["l3_style_agent", "l3_content_plan_agent"],
            "l3_preference_adapter",
        )
        surface_builder.add_edge(
            ["l3_preference_adapter", "l3_visual_agent"],
            "l4_surface_directive_collector",
        )
        surface_builder.add_edge("l4_surface_directive_collector", END)

        surface_graph = surface_builder.compile()
        result = await surface_graph.ainvoke(dict(initial_state))
    finally:
        _reset_service_bindings(service_bindings)
    return result


def _inject_surface_services(
    services: CognitionChainServices,
) -> list[ServiceBinding]:
    """Attach caller-provided LLM instances to moved L3 stage modules."""

    service_bindings = [
        (reset_json_parser, set_json_parser(services.parse_json)),
        (l3.reset_style_agent_llm, l3.set_style_agent_llm(
            LLMStageBinding(services.llm, services.style_config),
        )),
        (l3.reset_content_plan_agent_llm, l3.set_content_plan_agent_llm(
            LLMStageBinding(services.llm, services.content_plan_config),
        )),
        (l3.reset_preference_adapter_llm, l3.set_preference_adapter_llm(
            LLMStageBinding(services.llm, services.preference_config),
        )),
        (l3.reset_visual_agent_llm, l3.set_visual_agent_llm(
            LLMStageBinding(services.llm, services.visual_config),
        )),
    ]
    return service_bindings


def _reset_service_bindings(service_bindings: list[ServiceBinding]) -> None:
    """Restore previous service bindings after one surface graph run."""

    for reset_binding, token in reversed(service_bindings):
        reset_binding(token)


def _state_from_surface_input(
    input_payload: CognitionTextSurfaceInputV1,
) -> dict[str, Any]:
    """Build the internal L3 state from the selected-surface contract."""

    chain_input = input_payload["chain_input"]
    character = chain_input["character"]
    current_user = chain_input["current_user"]
    current_event = chain_input["current_event"]
    scene = chain_input["scene"]
    context = chain_input["conversation_context"]
    evidence = chain_input["evidence"]
    residue = input_payload["cognition_residue"]
    intent = input_payload["selected_text_surface_intent"]
    state = {
        "llm_trace_id": chain_input.get("llm_trace_id", ""),
        "character_profile": {
            "global_user_id": character["character_global_id"],
            "name": character["name"],
            "description": character["description"],
            "gender": character["gender"],
            "age": character["age"],
            "birthday": character["birthday"],
            "background": character["backstory"],
            "personality_brief": _mapping(character["personality_brief"]),
            "boundary_profile": _mapping(character["boundary_profile"]),
            "linguistic_texture_profile": _mapping(
                character["linguistic_texture_profile"],
            ),
            "mood": character["mood"],
            "global_vibe": character["global_vibe"],
        },
        "storage_timestamp_utc": scene["storage_timestamp_utc"],
        "local_time_context": _mapping(scene["local_time_context"]),
        "user_input": current_event["user_input"],
        "prompt_message_context": _mapping(
            current_event.get("prompt_message_context", {}),
        ),
        "platform": scene["platform"],
        "platform_channel_id": "",
        "channel_type": scene["channel_type"],
        "global_user_id": current_user["global_user_id"],
        "user_name": current_user["display_name"],
        "user_profile": {
            **_mapping(current_user.get("profile", {})),
            "global_user_id": current_user["global_user_id"],
            "user_name": current_user["display_name"],
            "affinity": current_user["affinity"],
            "affinity_level": current_user["affinity_level"],
            "last_relationship_insight": (
                current_user["last_relationship_insight"]
            ),
        },
        "platform_bot_id": "",
        "chat_history_recent": _mapping_list(scene["interaction_history_recent"]),
        "reply_context": _mapping(current_event.get("reply_context", {})),
        "indirect_speech_context": current_event["indirect_speech_context"],
        "channel_topic": scene["channel_topic"],
        "conversation_progress": _mapping(context["conversation_progress"]),
        "promoted_reflection_context": _mapping(
            context["promoted_reflection_context"],
        ),
        "decontexualized_input": current_event["decontextualized_input"],
        "referents": _mapping_list(current_event["referents"]),
        "rag_result": _rag_result_from_evidence(evidence, current_user),
        "emotional_appraisal": residue["emotional_appraisal"],
        "interaction_subtext": residue["interaction_subtext"],
        "internal_monologue": residue["internal_monologue"],
        "character_intent": residue["character_intent"],
        "logical_stance": residue["logical_stance"],
        "judgment_note": residue["judgment_note"],
        "boundary_core_assessment": _boundary_assessment_from_residue(residue),
        "social_distance": residue["social_distance"],
        "emotional_intensity": residue["emotional_intensity"],
        "vibe_check": residue["vibe_check"],
        "relational_dynamic": residue["relational_dynamic"],
        "selected_text_surface_intent": intent["speak_intent"],
        "memory_lifecycle_context": input_payload["memory_lifecycle_context"],
        "interaction_style_context": input_payload["interaction_style_context"],
        "cognitive_episode": build_prompt_selection_episode(chain_input),
        "coding_run_followup": _mapping(
            chain_input.get("coding_run_followup", {}),
        ),
    }
    resolver = chain_input["resolver"]
    resolver_state = resolver.get("resolver_state")
    if isinstance(resolver_state, Mapping):
        state["resolver_state"] = dict(resolver_state)
    resolver_goal_progress = resolver["goal_progress"]
    if isinstance(resolver_goal_progress, Mapping):
        state["resolver_goal_progress"] = dict(resolver_goal_progress)
    return state


def _rag_result_from_evidence(
    evidence: Mapping[str, Any],
    current_user: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the prompt-safe RAG payload supplied by the caller."""

    raw_rag_result = evidence.get("rag_result")
    if isinstance(raw_rag_result, Mapping):
        return_value = dict(raw_rag_result)
        return_value.setdefault("answer", evidence["rag_answer"])
        return_value.setdefault(
            "current_user_rag_bundle",
            evidence["current_user_rag_bundle"],
        )
        return_value.setdefault(
            "user_image",
            _user_image_from_current_user(current_user),
        )
        return return_value
    return_value = {
        "answer": evidence["rag_answer"],
        "memory_evidence": evidence["memory_evidence"],
        "current_user_rag_bundle": evidence["current_user_rag_bundle"],
        "user_image": _user_image_from_current_user(current_user),
    }
    return return_value


def _user_image_from_current_user(
    current_user: Mapping[str, Any],
) -> dict[str, Any]:
    """Project public current-user memory summaries into the legacy RAG slot."""

    memory_context = _mapping(current_user.get("memory_context", {}))
    user_memory_context = empty_user_memory_context()
    user_memory_context.update({
        "durable_profile_summary": memory_context.get(
            "durable_profile_summary",
            "",
        ),
        "relationship_summary": memory_context.get("relationship_summary", ""),
        "recent_commitments_summary": memory_context.get(
            "recent_commitments_summary",
            "",
        ),
        "known_preferences_summary": memory_context.get(
            "known_preferences_summary",
            "",
        ),
    })
    return_value = {
        "user_memory_context": user_memory_context,
    }
    return return_value


def _boundary_assessment_from_residue(
    residue: Mapping[str, Any],
) -> dict[str, str]:
    """Build the prompt-safe boundary summary needed by visual planning."""

    return_value = {
        "boundary_issue": "",
        "boundary_summary": str(residue.get("judgment_note", "")),
        "behavior_primary": "",
        "behavior_secondary": "none",
        "acceptance": "allow",
        "stance_bias": str(residue.get("logical_stance", "")),
        "identity_policy": "",
        "pressure_policy": "",
        "trajectory": "",
    }
    return return_value


def _mapping(value: object) -> dict[str, Any]:
    """Return a plain mapping copy or an empty dict."""

    if isinstance(value, Mapping):
        return_value = dict(value)
        return return_value
    return_value: dict[str, Any] = {}
    return return_value


def _mapping_list(value: object) -> list[dict[str, Any]]:
    """Return a list containing only mapping rows."""

    if not isinstance(value, list):
        return_value: list[dict[str, Any]] = []
        return return_value
    rows = [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]
    return rows
