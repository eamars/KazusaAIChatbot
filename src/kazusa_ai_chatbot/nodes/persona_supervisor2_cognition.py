"""Cognition subgraph — state definition, graph wiring, and entry-point.

Agent implementations live in the layer-specific submodules:
  - persona_supervisor2_cognition_l1  (L1 subconscious)
  - persona_supervisor2_cognition_l2  (L2 consciousness / boundary / judgment)
  - persona_supervisor2_cognition_l3  (L3 contextual / linguistic / visual + L4 collector)
"""
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState, CognitionState
from kazusa_ai_chatbot.utils import build_interaction_history_recent, log_preview

from langgraph.graph import StateGraph, START, END

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import agent functions from layer submodules
# ---------------------------------------------------------------------------
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l1 import (  # noqa: E402
    call_cognition_subconscious,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2 import (  # noqa: E402
    call_cognition_consciousness,
    call_boundary_core_agent,
    call_judgment_core_agent,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l3 import (  # noqa: E402
    call_contextual_agent,
    call_style_agent,
    call_content_anchor_agent,
    call_preference_adapter,
    call_visual_agent,
    call_collector,
)


async def call_cognition_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    """
    Future development plans: 
    
    - Separate the global character mood with the user specific mood. 
      * Global mood get update from all users' conversation
      * User mood get update from this user's conversation (this is not affinity. The mood can change indenpendently from affinity in time)

    """
    sub_agent_builder = StateGraph(CognitionState)

    sub_agent_builder.add_node("l1_subconscious", call_cognition_subconscious)
    sub_agent_builder.add_node("l2a_consciousness", call_cognition_consciousness)
    sub_agent_builder.add_node("l2b_boundary_core", call_boundary_core_agent)
    sub_agent_builder.add_node("l2c_judgment_core", call_judgment_core_agent)

    sub_agent_builder.add_node("l3_contextual_agent", call_contextual_agent)
    sub_agent_builder.add_node("l3_style_agent", call_style_agent)
    sub_agent_builder.add_node("l3_content_anchor_agent", call_content_anchor_agent)
    sub_agent_builder.add_node("l3_preference_adapter", call_preference_adapter)
    sub_agent_builder.add_node("l3_visual_agent", call_visual_agent)
    sub_agent_builder.add_node("l4_collector", call_collector)

    # Connect
    sub_agent_builder.add_edge(START, "l1_subconscious")
    sub_agent_builder.add_edge("l1_subconscious", "l2a_consciousness")
    sub_agent_builder.add_edge("l1_subconscious", "l2b_boundary_core")

    sub_agent_builder.add_edge("l2a_consciousness", "l2c_judgment_core")
    sub_agent_builder.add_edge("l2b_boundary_core", "l2c_judgment_core")

    sub_agent_builder.add_edge("l2c_judgment_core", "l3_contextual_agent")
    sub_agent_builder.add_edge("l2c_judgment_core", "l3_style_agent")
    sub_agent_builder.add_edge("l2c_judgment_core", "l3_content_anchor_agent")
    sub_agent_builder.add_edge("l2c_judgment_core", "l3_visual_agent")

    sub_agent_builder.add_edge("l3_contextual_agent", "l4_collector")
    sub_agent_builder.add_edge("l3_style_agent", "l3_preference_adapter")
    sub_agent_builder.add_edge("l3_preference_adapter", "l4_collector")
    sub_agent_builder.add_edge("l3_content_anchor_agent", "l4_collector")
    sub_agent_builder.add_edge("l3_visual_agent", "l4_collector")

    sub_agent_builder.add_edge("l4_collector", END)


    cognition_subgraph = sub_agent_builder.compile()

    # Get attributes
    decontexualized_input = state["decontexualized_input"]
    interaction_history_recent = build_interaction_history_recent(
        state["chat_history_wide"],
        state["platform_user_id"],
        state["platform_bot_id"],
        state["global_user_id"],
    )

    initial_state: CognitionState = {
        "character_profile": state["character_profile"],
        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "prompt_message_context": state["prompt_message_context"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history_recent": interaction_history_recent,
        "reply_context": state["reply_context"],
        "indirect_speech_context": state["indirect_speech_context"],
        "channel_topic": state["channel_topic"],
        "conversation_progress": state.get("conversation_progress"),

        # From previous stages
        "decontexualized_input": decontexualized_input,
        "referents": state["referents"],
        "rag_result": state["rag_result"],
    }
    
    result = await cognition_subgraph.ainvoke(initial_state)

    # Generate outputs
    internal_monologue = result.get("internal_monologue", "")
    action_directives = result.get("action_directives", {})
    interaction_subtext = result.get("interaction_subtext", "")
    emotional_appraisal = result.get("emotional_appraisal", "")
    character_intent = result.get("character_intent", "")
    logical_stance = result.get("logical_stance", "")

    logger.info(
        f"Cognition output: stance={logical_stance} "
        f"intent={character_intent} "
        f"appraisal={log_preview(emotional_appraisal)} "
        f"subtext={log_preview(interaction_subtext)} "
        f"action_directives={log_preview(action_directives)} "
        f"monologue={log_preview(internal_monologue)}"
    )
    logger.debug(
        f'Cognition input: input={log_preview(state["decontexualized_input"])}'
    )


    return_value = {
        "internal_monologue": internal_monologue,
        "action_directives": action_directives,

        # Other data used by post-dialog consolidation.
        "interaction_subtext": interaction_subtext,
        "emotional_appraisal": emotional_appraisal,
        "character_intent": character_intent,
        "logical_stance": logical_stance,
    }
    return return_value
