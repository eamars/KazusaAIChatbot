"""Public main-chain entrypoint and internal cognition graph wiring."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.cognition_chain_core.contracts import (
    CognitionChainInputV1,
    CognitionChainOutputV1,
    CognitionChainServices,
    LLMStageBinding,
    validate_cognition_chain_input,
    validate_cognition_chain_output,
)
from kazusa_ai_chatbot.cognition_chain_core.episode_projection import (
    build_prompt_selection_episode,
)
from kazusa_ai_chatbot.cognition_chain_core.graph_state import CoreStageState
from kazusa_ai_chatbot.cognition_chain_core.stages import l1, l2, l2c2, l2d
from kazusa_ai_chatbot.cognition_chain_core.utils import (
    empty_user_memory_context,
    reset_json_parser,
    set_json_parser,
)

ServiceBinding = tuple[Callable[[Any], None], Any]


async def run_cognition_chain(
    input_payload: CognitionChainInputV1,
    services: CognitionChainServices,
) -> CognitionChainOutputV1:
    """Run one reusable cognition-chain pass from the public input contract."""

    validated_input = validate_cognition_chain_input(input_payload)
    initial_state = _state_from_chain_input(validated_input)
    graph_result = await run_cognition_state_graph(initial_state, services)
    output = _chain_output_from_graph_result(graph_result)
    validated_output = validate_cognition_chain_output(output)
    return validated_output


async def run_cognition_state_graph(
    initial_state: Mapping[str, Any],
    services: CognitionChainServices,
) -> dict[str, Any]:
    """Run the moved stage graph over an already projected stage state."""

    service_bindings = _inject_services(services)

    try:
        sub_agent_builder = StateGraph(CoreStageState)

        sub_agent_builder.add_node(
            "l1_subconscious",
            l1.call_cognition_subconscious,
        )
        sub_agent_builder.add_node(
            "l2a_conscious_framing",
            l2.call_cognition_consciousness,
        )
        sub_agent_builder.add_node(
            "l2b_boundary_appraisal",
            l2.call_boundary_core_agent,
        )
        sub_agent_builder.add_node(
            "l2c1_judgment_synthesis",
            l2.call_judgment_core_agent,
        )
        sub_agent_builder.add_node(
            "l2c2_social_context_appraisal",
            l2c2.call_social_context_appraisal,
        )
        sub_agent_builder.add_node(
            "l2d_action_selection",
            l2d.select_semantic_actions,
        )

        sub_agent_builder.add_edge(START, "l1_subconscious")
        sub_agent_builder.add_edge("l1_subconscious", "l2a_conscious_framing")
        sub_agent_builder.add_edge("l1_subconscious", "l2b_boundary_appraisal")
        sub_agent_builder.add_edge(
            ["l2a_conscious_framing", "l2b_boundary_appraisal"],
            "l2c1_judgment_synthesis",
        )
        sub_agent_builder.add_edge(
            "l2b_boundary_appraisal",
            "l2c2_social_context_appraisal",
        )
        sub_agent_builder.add_edge(
            ["l2c1_judgment_synthesis", "l2c2_social_context_appraisal"],
            "l2d_action_selection",
        )
        sub_agent_builder.add_edge("l2d_action_selection", END)

        cognition_subgraph = sub_agent_builder.compile()
        result = await cognition_subgraph.ainvoke(dict(initial_state))
    finally:
        _reset_service_bindings(service_bindings)
    return result


def _inject_services(services: CognitionChainServices) -> list[ServiceBinding]:
    """Attach caller-provided LLM instances to moved stage modules."""

    service_bindings = [
        (reset_json_parser, set_json_parser(services.parse_json)),
        (l1.reset_subconscious_llm, l1.set_subconscious_llm(
            LLMStageBinding(services.llm, services.cognition_config),
        )),
        (l2.reset_conscious_llm, l2.set_conscious_llm(
            LLMStageBinding(services.llm, services.cognition_config),
        )),
        (l2.reset_boundary_core_llm, l2.set_boundary_core_llm(
            LLMStageBinding(services.llm, services.boundary_core_config),
        )),
        (l2.reset_judgement_core_llm, l2.set_judgement_core_llm(
            LLMStageBinding(services.llm, services.cognition_config),
        )),
        (l2c2.reset_contextual_agent_llm, l2c2.set_contextual_agent_llm(
            LLMStageBinding(services.llm, services.cognition_config),
        )),
        (l2d.reset_action_selection_llm, l2d.set_action_selection_llm(
            LLMStageBinding(services.llm, services.action_selection_config),
        )),
    ]
    return service_bindings


def _reset_service_bindings(service_bindings: list[ServiceBinding]) -> None:
    """Restore previous service bindings after one graph run."""

    for reset_binding, token in reversed(service_bindings):
        reset_binding(token)


def _state_from_chain_input(
    input_payload: CognitionChainInputV1,
) -> dict[str, Any]:
    """Build the internal stage state from the public semantic contract."""

    character = input_payload["character"]
    current_user = input_payload["current_user"]
    current_event = input_payload["current_event"]
    scene = input_payload["scene"]
    context = input_payload["conversation_context"]
    evidence = input_payload["evidence"]
    resolver = input_payload["resolver"]
    rag_result = _rag_result_from_evidence(evidence, current_user)
    character_profile = {
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
    }
    user_profile = dict(_mapping(current_user.get("profile", {})))
    user_profile.update({
        "global_user_id": current_user["global_user_id"],
        "user_name": current_user["display_name"],
        "affinity": current_user["affinity"],
        "affinity_level": current_user["affinity_level"],
        "last_relationship_insight": current_user["last_relationship_insight"],
    })
    action_selection_context = _mapping(input_payload["action_selection_context"])
    state = {
        "llm_trace_id": input_payload.get("llm_trace_id", ""),
        "character_profile": character_profile,
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
        "user_profile": user_profile,
        "platform_bot_id": "",
        "chat_history_recent": _mapping_list(scene["interaction_history_recent"]),
        "reply_context": _mapping(current_event.get("reply_context", {})),
        "indirect_speech_context": current_event["indirect_speech_context"],
        "channel_topic": scene["channel_topic"],
        "conversation_progress": _mapping(context["conversation_progress"]),
        "promoted_reflection_context": _mapping(
            context["promoted_reflection_context"],
        ),
        "internal_monologue_residue_context": (
            context["internal_monologue_residue_context"]
        ),
        "past_dialog_cognition_context": context.get(
            "past_dialog_cognition_context",
            "",
        ),
        "decontexualized_input": current_event["decontextualized_input"],
        "referents": _mapping_list(current_event["referents"]),
        "rag_result": rag_result,
        "resolver_context": resolver["resolver_context"],
        "cognitive_episode": build_prompt_selection_episode(input_payload),
        "coding_run_followup": _mapping(
            input_payload.get("coding_run_followup", {}),
        ),
        "action_selection_context": dict(action_selection_context),
        "available_action_affordances": list(input_payload["available_actions"]),
        "max_action_requests": (
            input_payload["runtime_context"]["max_action_requests"]
        ),
        "max_resolver_requests": (
            input_payload["runtime_context"]["max_resolver_requests"]
        ),
        "background_work_output_char_limit": (
            input_payload["runtime_context"]["background_work_output_char_limit"]
        ),
        "task_willingness_boundary_enabled": (
            input_payload["runtime_context"]["task_willingness_boundary_enabled"]
        ),
    }
    group_engagement_context = action_selection_context.get(
        "group_engagement_action_context",
    )
    if isinstance(group_engagement_context, Mapping):
        state["group_engagement_action_context"] = dict(group_engagement_context)
    pending_resume = resolver["pending_resume"]
    if isinstance(pending_resume, Mapping):
        state["pending_resolver_resume"] = dict(pending_resume)
    goal_progress = resolver["goal_progress"]
    if isinstance(goal_progress, Mapping):
        state["resolver_goal_progress"] = dict(goal_progress)
    resolver_state = resolver.get("resolver_state")
    if isinstance(resolver_state, Mapping):
        state["resolver_state"] = dict(resolver_state)
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
    """Project the public current-user memory block into the legacy RAG slot."""

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


def _chain_output_from_graph_result(
    result: Mapping[str, Any],
) -> CognitionChainOutputV1:
    """Project moved graph output into the public core output contract."""

    output: CognitionChainOutputV1 = {
        "schema_version": "cognition_chain_output.v1",
        "cognition_residue": {
            "emotional_appraisal": str(result.get("emotional_appraisal", "")),
            "interaction_subtext": str(result.get("interaction_subtext", "")),
            "internal_monologue": str(result.get("internal_monologue", "")),
            "logical_stance": str(result.get("logical_stance", "")),
            "character_intent": str(result.get("character_intent", "")),
            "judgment_note": str(result.get("judgment_note", "")),
            "social_distance": str(result.get("social_distance", "")),
            "emotional_intensity": str(result.get("emotional_intensity", "")),
            "vibe_check": str(result.get("vibe_check", "")),
            "relational_dynamic": str(result.get("relational_dynamic", "")),
        },
        "semantic_action_requests": list(
            result.get("semantic_action_requests", [])
        ),
        "resolver_capability_requests": list(
            result.get("resolver_capability_requests", [])
        ),
        "chain_trace": {
            "stage_order": ["l1", "l2a", "l2b", "l2c1", "l2c2", "l2d"],
            "selected_actions_summary": "",
            "resolver_summary": "",
            "warnings": [],
        },
    }
    resolver_pending_resolution = result.get("resolver_pending_resolution")
    if isinstance(resolver_pending_resolution, dict):
        output["resolver_pending_resolution"] = resolver_pending_resolution
    resolver_goal_progress = result.get("resolver_goal_progress")
    if isinstance(resolver_goal_progress, dict):
        output["resolver_goal_progress"] = resolver_goal_progress
    return output
