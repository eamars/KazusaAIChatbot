from langgraph.graph import StateGraph, START, END

from kazusa_ai_chatbot.mcp_client import mcp_manager

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
        "platform_user_id": state["platform_user_id"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history": state["chat_history"],
        "user_topic": state["user_topic"],
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


async def test_main():
    from kazusa_ai_chatbot.db import get_character_profile, get_conversation_history, get_user_profile
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.utils import load_personality
    import datetime

    # Connect to MCP tool servers
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — tools will be unavailable")

    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=10)
    trimmed_history = trim_history_dict(history)

    # Create a mocked BotState
    test_state: IMProcessState = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "platform": "discord",
        "platform_user_id": "320899931776745483",
        "global_user_id": "test-uuid-placeholder",
        "user_name": "EAMARS",
        "user_input": "既然作业已经写完了，千纱准备'奖励'我了么♥",
        "user_profile": await get_user_profile("test-uuid-placeholder"),

        "platform_bot_id": "1485169644888395817",
        "bot_name": "KazusaBot",
        "character_profile": await get_character_profile(),

        "platform_channel_id": "",
        "channel_name": "",
        "chat_history": trimmed_history,

        "should_respond": True,
        "reason_to_respond": "User is asking a question",
        "use_reply_feature": False,
        "channel_topic": "General chat",
        "user_topic": "作业交流",
    }
    
    result = await persona_supervisor2(test_state)
    print(result)

    await mcp_manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
