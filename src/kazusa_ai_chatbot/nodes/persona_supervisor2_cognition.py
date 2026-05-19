"""Cognition subgraph — state definition, graph wiring, and entry-point.

Agent implementations live in the layer-specific submodules:
  - persona_supervisor2_cognition_l1  (L1 subconscious)
  - persona_supervisor2_cognition_l2   (L2a/L2b/L2c1 cognition)
  - persona_supervisor2_cognition_l2c2 (L2c2 social context)
  - persona_supervisor2_cognition_l2d  (L2d action selection)
"""
from collections.abc import Mapping
import logging

from langgraph.graph import StateGraph, START, END

from kazusa_ai_chatbot.db import build_group_engagement_action_context
from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState, CognitionState
from kazusa_ai_chatbot.utils import build_interaction_history_recent, log_preview

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
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2c2 import (  # noqa: E402
    call_social_context_appraisal,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_cognition_l2d import (  # noqa: E402
    call_action_initializer,
)


async def call_group_engagement_action_context_loader(
    state: CognitionState,
) -> CognitionState:
    """Load group-channel engagement guidance before action selection."""

    empty_context = {"engagement_guidelines": [], "confidence": ""}
    if not _is_group_self_cognition_state(state):
        return_value = {
            "group_engagement_action_context": empty_context,
        }
        return return_value

    context = await build_group_engagement_action_context(
        channel_type=state["channel_type"],
        platform=state["platform"],
        platform_channel_id=state["platform_channel_id"],
    )
    return_value = {
        "group_engagement_action_context": context,
    }
    return return_value


def _is_group_self_cognition_state(state: CognitionState) -> bool:
    """Return whether the current cognition state is group self-cognition."""

    if state["channel_type"] != "group":
        return_value = False
        return return_value

    episode = state.get("cognitive_episode")
    if not isinstance(episode, Mapping):
        return_value = False
        return return_value

    trigger_source = episode.get("trigger_source")
    input_sources = episode.get("input_sources")
    is_internal_monologue = (
        isinstance(input_sources, list)
        and "internal_monologue" in input_sources
    )
    is_self_cognition = (
        trigger_source == "internal_thought"
        and is_internal_monologue
    )
    return is_self_cognition


async def call_cognition_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    """
    Future development plans: 
    
    - Separate the global character mood with the user specific mood. 
      * Global mood get update from all users' conversation
      * User mood get update from this user's conversation (this is not affinity. The mood can change indenpendently from affinity in time)

    """
    sub_agent_builder = StateGraph(CognitionState)

    sub_agent_builder.add_node("l1_subconscious", call_cognition_subconscious)
    sub_agent_builder.add_node(
        "l2a_conscious_framing",
        call_cognition_consciousness,
    )
    sub_agent_builder.add_node("l2b_boundary_appraisal", call_boundary_core_agent)
    sub_agent_builder.add_node("l2c1_judgment_synthesis", call_judgment_core_agent)
    sub_agent_builder.add_node(
        "l2c2_social_context_appraisal",
        call_social_context_appraisal,
    )
    sub_agent_builder.add_node(
        "group_engagement_action_context",
        call_group_engagement_action_context_loader,
    )
    sub_agent_builder.add_node("l2d_action_selection", call_action_initializer)

    # Connect
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
        "group_engagement_action_context",
    )
    sub_agent_builder.add_edge(
        "group_engagement_action_context",
        "l2d_action_selection",
    )
    sub_agent_builder.add_edge("l2d_action_selection", END)

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

        # From previous stages
        "decontexualized_input": decontexualized_input,
        "referents": state["referents"],
        "rag_result": state["rag_result"],
    }
    cognitive_episode = state.get("cognitive_episode")
    if cognitive_episode is not None:
        initial_state["cognitive_episode"] = cognitive_episode
    
    result = await cognition_subgraph.ainvoke(initial_state)

    # Generate outputs
    internal_monologue = result.get("internal_monologue", "")
    interaction_subtext = result.get("interaction_subtext", "")
    emotional_appraisal = result.get("emotional_appraisal", "")
    character_intent = result.get("character_intent", "")
    logical_stance = result.get("logical_stance", "")
    judgment_note = result.get("judgment_note", "")
    social_distance = result["social_distance"]
    emotional_intensity = result["emotional_intensity"]
    vibe_check = result["vibe_check"]
    relational_dynamic = result["relational_dynamic"]
    action_specs = result.get("action_specs", [])

    logger.info(
        f"Cognition output: stance={logical_stance} "
        f"intent={character_intent} "
        f"appraisal={log_preview(emotional_appraisal)} "
        f"subtext={log_preview(interaction_subtext)} "
        f"action_specs={log_preview(action_specs)} "
        f"monologue={log_preview(internal_monologue)}"
    )
    logger.debug(
        f'Cognition input: input={log_preview(state["decontexualized_input"])}'
    )


    return_value = {
        "internal_monologue": internal_monologue,
        "action_specs": action_specs,

        # Other data used by post-dialog consolidation.
        "interaction_subtext": interaction_subtext,
        "emotional_appraisal": emotional_appraisal,
        "character_intent": character_intent,
        "logical_stance": logical_stance,
        "judgment_note": judgment_note,
        "social_distance": social_distance,
        "emotional_intensity": emotional_intensity,
        "vibe_check": vibe_check,
        "relational_dynamic": relational_dynamic,
    }
    return return_value
