from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from openai.types.shared import reasoning

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_PERSONA_SUPERVISOR_STAGE1_RETRY
from kazusa_ai_chatbot.state import AgentResult, BotState
from kazusa_ai_chatbot.utils import parse_llm_json_output

from kazusa_ai_chatbot.mcp_client import mcp_manager

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer
from kazusa_ai_chatbot.nodes.persona_supervisor2_research_subgraph import call_research_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import call_cognition_subgraph

from typing import Annotated, TypedDict, List
import json
import logging



logger = logging.getLogger(__name__)


_llm: ChatOpenAI | None = None
def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.5,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


async def call_expression_subgraph(state: GlobalPersonaState) -> dict:
    pass

async def call_consolidation_subgraph(state: GlobalPersonaState) -> dict:
    pass


async def persona_supervisor2(state: BotState) -> dict:

    # Build the top level graph that connect stages
    persona_builder = StateGraph(GlobalPersonaState)
    persona_builder.add_node("stage_0_msg_decontexualizer", call_msg_decontexualizer)
    persona_builder.add_node("stage_1_research", call_research_subgraph)
    persona_builder.add_node("stage_2_cognition", call_cognition_subgraph)
    persona_builder.add_node("stage_3_expression", call_expression_subgraph)
    persona_builder.add_node("stage_4_consolidation", call_consolidation_subgraph)

    # Build linear flow
    persona_builder.add_edge(START, "stage_0_msg_decontexualizer")
    persona_builder.add_edge("stage_0_msg_decontexualizer", "stage_1_research")
    persona_builder.add_edge("stage_1_research", "stage_2_cognition")
    persona_builder.add_edge("stage_2_cognition", "stage_3_expression")
    persona_builder.add_edge("stage_3_expression", "stage_4_consolidation")
    persona_builder.add_edge("stage_4_consolidation", END)

    
    persona_graph = persona_builder.compile()

    initial_persona_state: GlobalPersonaState = {
        # Character Related
        "character_state": state["character_state"],
        "character_profile": state["personality"],

        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["message_text"],
        "user_id": state.get("user_id", ""),
        "user_name": state.get("user_name", ""),
        "user_affinity_score": state.get("affinity", AFFINITY_DEFAULT),  
        "bot_id": state.get("bot_id", ""),
        "chat_history": state.get("chat_history", []),
        "user_topic": state.get("assembler_output", {}).get("user_topic", ""),
        "channel_topic": state.get("assembler_output", {}).get("channel_topic", ""),
    }
    
    results = await persona_graph.ainvoke(initial_persona_state)
    
    return results


async def test_main():
    from kazusa_ai_chatbot.db import AFFINITY_DEFAULT, get_affinity, get_character_state, get_conversation_history, get_user_facts
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.utils import load_personality
    import datetime

    # Connect to MCP tool servers
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — tools will be unavailable")

    history = await get_conversation_history(channel_id="1485606207069880361", limit=10)
    trimmed_history = trim_history_dict(history)

    # Create a mocked BotState
    test_state: BotState = {
        "personality": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state(),
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message_text": "1+1=？",
        "user_id": "320899931776745483",
        "user_name": "EAMARS",
        "affinity": 1000,
        "bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "assembler_output": {
            "channel_topic": "课间交流",
            "user_topic": "交流作业"
        }
    }
    
    result = await persona_supervisor2(test_state)
    print(result)

    await mcp_manager.stop()

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
