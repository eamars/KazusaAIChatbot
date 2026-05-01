"""Persona graph orchestration for decontextualization, RAG, cognition, and dialog."""

import logging

from langgraph.graph import END, START, StateGraph

from kazusa_ai_chatbot.config import CHAT_HISTORY_RECENT_LIMIT
from kazusa_ai_chatbot.nodes.dialog_agent import dialog_agent
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition import call_cognition_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_consolidator import call_consolidation_subgraph
from kazusa_ai_chatbot.nodes.persona_supervisor2_msg_decontexualizer import call_msg_decontexualizer
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_projection import project_known_facts
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import call_rag_supervisor
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.nodes.referent_resolution import (
    should_skip_rag_for_unresolved_referents,
    unresolved_referent_reason,
)
from kazusa_ai_chatbot.state import IMProcessState
from kazusa_ai_chatbot.utils import build_interaction_history_recent, log_preview

logger = logging.getLogger(__name__)


async def call_action_subgraph(state: GlobalPersonaState) -> dict:
    """Run dialog generation and attach deterministic response addressing.

    Args:
        state: Current persona graph state.

    Returns:
        Partial state update with dialog fragments and addressed users.
    """

    result = await dialog_agent(state)
    final_dialog = result["final_dialog"]
    return_value = {
        "final_dialog": final_dialog,
        "target_addressed_user_ids": result["target_addressed_user_ids"],
        "target_broadcast": result["target_broadcast"],
    }
    return return_value


async def stage_1_research(state: GlobalPersonaState) -> dict:
    """Run RAG2 and project its facts into the persona payload.

    Args:
        state: Current top-level persona graph state.

    Returns:
        A partial state update containing the projected ``rag_result``.
    """
    referents = state["referents"]
    if should_skip_rag_for_unresolved_referents(referents):
        referent_reason = unresolved_referent_reason(referents)
        rag_result = project_known_facts(
            [],
            current_user_id=state["global_user_id"],
            character_user_id=state["character_profile"]["global_user_id"],
            answer="",
            unknown_slots=[],
            loop_count=0,
        )
        logger.info(f'RAG2 skipped for unresolved reference: platform={state["platform"]} channel={state["platform_channel_id"] or "<dm>"} user={state["global_user_id"]} query={log_preview(state["decontexualized_input"])} reason={log_preview(referent_reason)} rag_result={log_preview(rag_result)}')
        return_value = {
            "rag_result": rag_result,
        }
        return return_value

    rag_supervisor_result = await call_rag_supervisor(
        original_query=state["decontexualized_input"],
        character_name=state["character_profile"]["name"],
        context={
            "platform": state["platform"],
            "platform_channel_id": state["platform_channel_id"],
            "channel_type": state["channel_type"],
            "global_user_id": state["global_user_id"],
            "user_name": state["user_name"],
            "user_profile": state["user_profile"],
            "current_timestamp": state["timestamp"],
            "prompt_message_context": state["prompt_message_context"],
            "channel_topic": state["channel_topic"],
            "chat_history_recent": state["chat_history_recent"],
            "chat_history_wide": state["chat_history_wide"],
            "reply_context": state["reply_context"],
            "indirect_speech_context": state["indirect_speech_context"],
        },
    )
    rag_result = project_known_facts(
        rag_supervisor_result["known_facts"],
        current_user_id=state["global_user_id"],
        character_user_id=state["character_profile"]["global_user_id"],
        answer=str(rag_supervisor_result["answer"]),
        unknown_slots=rag_supervisor_result["unknown_slots"],
        loop_count=int(rag_supervisor_result["loop_count"] or 0),
    )
    trace = rag_result["supervisor_trace"]
    logger.info(f'RAG2 projection: platform={state["platform"]} channel={state["platform_channel_id"] or "<dm>"} user={state["global_user_id"]} query={log_preview(state["decontexualized_input"])} answer={log_preview(rag_result["answer"])} dispatched={len(trace["dispatched"])} user_image={bool(rag_result["user_image"])} character_image={bool(rag_result["character_image"])} third_party_profiles={len(rag_result["third_party_profiles"])} memory_evidence={len(rag_result["memory_evidence"])} conversation_evidence={len(rag_result["conversation_evidence"])} external_evidence={len(rag_result["external_evidence"])} rag_result={log_preview(rag_result)}')
    return_value = {
        "rag_result": rag_result,
    }
    return return_value



async def persona_supervisor2(state: IMProcessState) -> dict:
    """Run persona reasoning with history scoped to the active user thread.

    Args:
        state: Top-level chat graph state after relevance gating.

    Returns:
        Dialog output and the persona-state snapshot used by background tasks.
    """

    # Build the top level graph that connect stages
    persona_builder = StateGraph(GlobalPersonaState)
    persona_builder.add_node("stage_0_msg_decontexualizer", call_msg_decontexualizer)
    persona_builder.add_node("stage_1_research", stage_1_research)
    persona_builder.add_node("stage_2_cognition", call_cognition_subgraph)
    persona_builder.add_node("stage_3_action", call_action_subgraph)  # perform action
    persona_builder.add_edge(START, "stage_0_msg_decontexualizer")
    persona_builder.add_edge("stage_0_msg_decontexualizer", "stage_1_research")
    persona_builder.add_edge("stage_1_research", "stage_2_cognition")
    persona_builder.add_edge("stage_2_cognition", "stage_3_action")
    persona_builder.add_edge("stage_3_action", END)

    
    persona_graph = persona_builder.compile()

    interaction_history_wide = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )
    interaction_history_recent = interaction_history_wide[-CHAT_HISTORY_RECENT_LIMIT:]

    initial_persona_state: GlobalPersonaState = {
        # Character Related
        "character_profile": state["character_profile"],

        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "platform": state["platform"],
        "platform_channel_id": state["platform_channel_id"],
        "channel_type": state["channel_type"],
        "platform_message_id": state["platform_message_id"],
        "platform_user_id": state["platform_user_id"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_wide": interaction_history_wide,
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "conversation_episode_state": state.get("conversation_episode_state"),
        "conversation_progress": state.get("conversation_progress"),
        "referents": [],
        "debug_modes": state["debug_modes"],
    }
    
    results = await persona_graph.ainvoke(initial_persona_state)
    
    return_value = {
        "final_dialog": results["final_dialog"],
        "target_addressed_user_ids": results["target_addressed_user_ids"],
        "target_broadcast": bool(results["target_broadcast"]),
        "future_promises": [],
        "consolidation_state": results,
    }
    return return_value
