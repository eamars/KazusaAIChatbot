from langgraph.graph import StateGraph, START, END

from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag import call_rag_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import call_cognition_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator import call_consolidation_subgraph
from kazusa_ai_chatbot.agents.dialog_agent import dialog_agent

import logging



logger = logging.getLogger(__name__)


async def call_action_subgraph(state: GlobalPersonaState) -> dict:
    # For now we will only keep dialog output. In the future we will add vocal, action and perhaps TaskDispatcher agent to handle other types of actions
    result = await dialog_agent(state)
    return {
        "final_dialog": result.get("final_dialog", []),
    }



async def persona_supervisor2(state: IMProcessState) -> dict:

    # Build the top level graph that connect stages
    persona_builder = StateGraph(GlobalPersonaState)
    persona_builder.add_node("stage_0_msg_decontexualizer", call_msg_decontexualizer)
    persona_builder.add_node("stage_1_research", call_rag_subgraph)
    persona_builder.add_node("stage_2_cognition", call_cognition_subgraph)
    persona_builder.add_node("stage_3_action", call_action_subgraph)  # perform action
    persona_builder.add_node("stage_4_consolidation", call_consolidation_subgraph)  # memory saving

    # Build flow with conditional edge for no_remember debug mode
    persona_builder.add_edge(START, "stage_0_msg_decontexualizer")
    persona_builder.add_edge("stage_0_msg_decontexualizer", "stage_1_research")
    persona_builder.add_edge("stage_1_research", "stage_2_cognition")
    persona_builder.add_edge("stage_2_cognition", "stage_3_action")

    def _route_after_action(state):
        debug = state.get("debug_modes") or {}
        if debug.get("no_remember"):
            logger.info("no_remember active — skipping consolidation (stage 4)")
            return "end"
        return "consolidate"

    persona_builder.add_conditional_edges(
        "stage_3_action",
        _route_after_action,
        {"consolidate": "stage_4_consolidation", "end": END},
    )
    persona_builder.add_edge("stage_4_consolidation", END)

    
    persona_graph = persona_builder.compile()

    initial_persona_state: GlobalPersonaState = {
        # Character Related
        "character_profile": state["character_profile"],

        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "platform": state["platform"],
        "platform_message_id": state["platform_message_id"],
        "platform_user_id": state["platform_user_id"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_wide": state["chat_history_wide"],
        "chat_history_recent": state["chat_history_recent"],
        "indirect_speech_context": state.get("indirect_speech_context", ""),
        "channel_topic": state["channel_topic"],
        "debug_modes": state.get("debug_modes", {}),
    }
    
    results = await persona_graph.ainvoke(initial_persona_state)
    
    return {
        "final_dialog": results.get("final_dialog", []),
        # "mood": results["mood"],
        # "global_vibe": results["global_vibe"],
        # "reflection_summary": results["reflection_summary"],
        # "diary_entry": results["diary_entry"],
        # "affinity_delta": results["affinity_delta"],
        # "last_relationship_insight": results["last_relationship_insight"],
        # "new_facts": results["new_facts"],
        "future_promises": results.get("future_promises", []),
    }
