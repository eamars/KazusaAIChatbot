"""Selected L3 text-surface handler."""

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (
    call_collector,
    call_content_anchor_agent,
    call_contextual_agent,
    call_interaction_style_context_loader,
    call_preference_adapter,
    call_style_agent,
    call_visual_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import (
    CognitionState,
    GlobalPersonaState,
)
from kazusa_ai_chatbot.utils import build_interaction_history_recent


async def call_l3_text_surface_handler(state: GlobalPersonaState) -> dict:
    """Run L3 directive generation for a selected text surface.

    Args:
        state: Persona graph state after L2d has selected a ``speak`` action.

    Returns:
        Partial state update containing collected text-surface directives.
    """

    surface_builder = StateGraph(CognitionState)
    surface_builder.add_node("l3_contextual_agent", call_contextual_agent)
    surface_builder.add_node(
        "l3_interaction_style_context_loader",
        call_interaction_style_context_loader,
    )
    surface_builder.add_node("l3_style_agent", call_style_agent)
    surface_builder.add_node("l3_content_anchor_agent", call_content_anchor_agent)
    surface_builder.add_node("l3_preference_adapter", call_preference_adapter)
    surface_builder.add_node("l3_visual_agent", call_visual_agent)
    surface_builder.add_node("l4_collector", call_collector)

    surface_builder.add_edge(START, "l3_contextual_agent")
    surface_builder.add_edge(START, "l3_interaction_style_context_loader")
    surface_builder.add_edge("l3_contextual_agent", "l3_visual_agent")
    surface_builder.add_edge(
        "l3_interaction_style_context_loader",
        "l3_content_anchor_agent",
    )
    surface_builder.add_edge("l3_content_anchor_agent", "l3_visual_agent")
    surface_builder.add_edge("l3_interaction_style_context_loader", "l3_style_agent")
    surface_builder.add_edge("l3_style_agent", "l3_preference_adapter")
    surface_builder.add_edge("l3_preference_adapter", "l4_collector")
    surface_builder.add_edge("l3_visual_agent", "l4_collector")
    surface_builder.add_edge("l4_collector", END)

    surface_graph = surface_builder.compile()
    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    initial_state: CognitionState = {
        "character_profile": state["character_profile"],
        "timestamp": state["timestamp"],
        "time_context": state["time_context"],
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
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_state["cognitive_episode"] = cognitive_episode

    result = await surface_graph.ainvoke(initial_state)
    return_value = {
        "action_directives": result["action_directives"],
    }
    return return_value
