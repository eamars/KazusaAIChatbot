"""Selected L3 text-surface handler."""

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.action_spec.registry import SPEAK_CAPABILITY
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    call_content_anchor_agent,
    call_interaction_style_context_loader,
    call_preference_adapter,
    call_surface_directive_collector,
    call_style_agent,
    call_visual_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.utils import build_interaction_history_recent


def _selected_text_surface_intent(state: GlobalPersonaState) -> str:
    """Project the selected text action into one model-facing intent string."""

    raw_specs = state.get("action_specs")
    if not isinstance(raw_specs, list):
        return_value = ""
        return return_value

    for action_spec in raw_specs:
        if not isinstance(action_spec, dict):
            continue
        kind = action_spec.get("kind")
        if kind != SPEAK_CAPABILITY:
            continue
        intent_parts = _text_surface_intent_parts(action_spec)
        return_value = "；".join(intent_parts)
        return return_value

    return_value = ""
    return return_value


def _text_surface_intent_parts(action_spec: dict) -> list[str]:
    """Extract prompt-safe text-surface requirements from one action spec."""

    params = action_spec.get("params")
    if not isinstance(params, dict):
        params = {}
    raw_requirements = params.get("surface_requirements")
    if not isinstance(raw_requirements, dict):
        raw_requirements = {}

    intent_parts: list[str] = []
    for field_name, label in (
        ("decision", "决策"),
        ("intent", "目标"),
        ("detail", "内容要求"),
        ("tone", "语气"),
    ):
        value = raw_requirements.get(field_name)
        if isinstance(value, str) and value.strip():
            intent_parts.append(f"{label}：{value.strip()}")

    reason = action_spec.get("reason")
    if isinstance(reason, str) and reason.strip():
        intent_parts.append(f"理由：{reason.strip()}")

    return intent_parts


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict:
    """Run L3 directive generation for a selected text surface.

    Args:
        state: Persona graph state after L2d has selected a ``speak`` action.

    Returns:
        Partial state update containing collected text-surface directives.
    """

    surface_builder = StateGraph(CognitionState)
    surface_builder.add_node(
        "l3_interaction_style_context_loader",
        call_interaction_style_context_loader,
    )
    surface_builder.add_node("l3_style_agent", call_style_agent)
    surface_builder.add_node("l3_content_anchor_agent", call_content_anchor_agent)
    surface_builder.add_node("l3_preference_adapter", call_preference_adapter)
    surface_builder.add_node("l3_visual_agent", call_visual_agent)
    surface_builder.add_node(
        "l4_surface_directive_collector",
        call_surface_directive_collector,
    )

    surface_builder.add_edge(START, "l3_interaction_style_context_loader")
    surface_builder.add_edge(
        "l3_interaction_style_context_loader",
        "l3_content_anchor_agent",
    )
    surface_builder.add_edge("l3_content_anchor_agent", "l3_visual_agent")
    surface_builder.add_edge("l3_interaction_style_context_loader", "l3_style_agent")
    surface_builder.add_edge("l3_style_agent", "l3_preference_adapter")
    surface_builder.add_edge(
        ["l3_preference_adapter", "l3_visual_agent"],
        "l4_surface_directive_collector",
    )
    surface_builder.add_edge("l4_surface_directive_collector", END)

    surface_graph = surface_builder.compile()
    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    initial_state: CognitionState = {
        "character_profile": state["character_profile"],
        "storage_timestamp_utc": state["storage_timestamp_utc"],
        "local_time_context": state["local_time_context"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "conversation_progress": state.get("conversation_progress"),
        "promoted_reflection_context": state.get("promoted_reflection_context"),
        "decontexualized_input": state["decontexualized_input"],
        "referents": state["referents"],
        "rag_result": state["rag_result"],
        "emotional_appraisal": state["emotional_appraisal"],
        "interaction_subtext": state["interaction_subtext"],
        "internal_monologue": state["internal_monologue"],
        "character_intent": state["character_intent"],
        "logical_stance": state["logical_stance"],
        "judgment_note": state["judgment_note"],
        "social_distance": state["social_distance"],
        "emotional_intensity": state["emotional_intensity"],
        "vibe_check": state["vibe_check"],
        "relational_dynamic": state["relational_dynamic"],
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_state["cognitive_episode"] = cognitive_episode
    selected_text_surface_intent = _selected_text_surface_intent(state)
    if selected_text_surface_intent:
        initial_state["selected_text_surface_intent"] = (
            selected_text_surface_intent
        )
    memory_lifecycle_context = state.get("memory_lifecycle_context")
    if isinstance(memory_lifecycle_context, dict):
        initial_state["memory_lifecycle_context"] = memory_lifecycle_context

    result = await surface_graph.ainvoke(initial_state)
    return_value = {
        "action_directives": result["action_directives"],
    }
    return return_value
