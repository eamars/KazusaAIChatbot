"""Cognition subgraph — state definition, graph wiring, and entry-point.

Agent implementations live in the layer-specific submodules:
  - persona_supervisor2_cognition_l1  (L1 subconscious)
  - persona_supervisor2_cognition_l2  (L2 consciousness / boundary / judgment)
  - persona_supervisor2_cognition_l3  (L3 contextual / linguistic / visual + L4 collector)
"""
from typing import TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.db import CharacterProfileDoc, get_character_profile, get_user_profile

from langgraph.graph import StateGraph, START, END

import logging

logger = logging.getLogger(__name__)


class CognitionState(TypedDict):
    character_profile: CharacterProfileDoc

    # Input from global state
    timestamp: str
    user_input: str
    global_user_id: str
    user_name: str
    user_profile: dict
    platform_bot_id: str
    chat_history: list[dict]
    user_topic: str
    channel_topic: str

    # Input from previous stage
    decontexualized_input: str
    research_facts: str

    # --- INTERNAL DATA FLOW ---
    # L1 Subconscious (L1) -> Conscious (L2)
    emotional_appraisal: str
    interaction_subtext: str

    # L2a Conscious (L2) -> (L3) -> evaluator (and output)
    internal_monologue: str
    character_intent: str
    logical_stance: str
    judgment_note: str

    # L2b Boundary core
    boundary_core_assessment: dict

    # L3 Has multiple parallel agents
    # L3 (Contextual Agent) Output
    social_distance: str
    emotional_intensity: str
    vibe_check: str
    relational_dynamic: str
    expression_willingness: str

    # L3 (Linguistic Agent) Output
    rhetorical_strategy: str
    linguistic_style: str
    content_anchors: list[str]
    forbidden_phrases: list[str]

    # L3 (Visual Agent) Output
    facial_expression: list[str]
    body_language: list[str]
    gaze_direction: list[str]
    visual_vibe: list[str]

    # L4 (Collector)
    action_directives: dict

    # --- CONTROL SIGNALS ---
    should_stop: bool
    reasoning: str
    retry: int


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
    call_linguistic_agent,
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
    sub_agent_builder.add_node("l3_linguistic_agent", call_linguistic_agent)
    sub_agent_builder.add_node("l3_visual_agent", call_visual_agent)
    sub_agent_builder.add_node("l4_collector", call_collector)

    # Connect
    sub_agent_builder.add_edge(START, "l1_subconscious")
    sub_agent_builder.add_edge("l1_subconscious", "l2a_consciousness")
    sub_agent_builder.add_edge("l1_subconscious", "l2b_boundary_core")

    sub_agent_builder.add_edge("l2a_consciousness", "l2c_judgment_core")
    sub_agent_builder.add_edge("l2b_boundary_core", "l2c_judgment_core")

    sub_agent_builder.add_edge("l2c_judgment_core", "l3_contextual_agent")
    sub_agent_builder.add_edge("l2c_judgment_core", "l3_linguistic_agent")
    sub_agent_builder.add_edge("l2c_judgment_core", "l3_visual_agent")

    sub_agent_builder.add_edge("l3_contextual_agent", "l4_collector")
    sub_agent_builder.add_edge("l3_linguistic_agent", "l4_collector")
    sub_agent_builder.add_edge("l3_visual_agent", "l4_collector")

    sub_agent_builder.add_edge("l4_collector", END)


    cognition_subgraph = sub_agent_builder.compile()

    # Get attributes
    decontexualized_input = state["decontexualized_input"]

    initial_state: CognitionState = {
        "character_profile": state["character_profile"],
        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "global_user_id": state["global_user_id"],
        "user_name": state["user_name"],
        "user_profile": state["user_profile"],
        "platform_bot_id": state["platform_bot_id"],
        "chat_history": state["chat_history"],
        "user_topic": state["user_topic"],
        "channel_topic": state["channel_topic"],

        # From previous stages
        "decontexualized_input": decontexualized_input,
        "research_facts": state["research_facts"],
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
        f"\nDecontexualized input: {state['decontexualized_input']}\n"
        f"  Internal monologue: {internal_monologue}\n"
        f"  Action directives: {action_directives}\n"
        f"  Interaction subtext: {interaction_subtext}\n"
        f"  Emotional appraisal: {emotional_appraisal}\n"
        f"  Character intent: {character_intent}\n"
        f"  Logical stance: {logical_stance}\n"
    )


    return {
        "internal_monologue": internal_monologue,
        "action_directives": action_directives,

        # Other data to help with stage 4 consolidation
        "interaction_subtext": interaction_subtext,
        "emotional_appraisal": emotional_appraisal,
        "character_intent": character_intent,
        "logical_stance": logical_stance,
    }


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history
    from kazusa_ai_chatbot.utils import load_personality


    history = await get_conversation_history(platform="discord", platform_channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    user_input = "既然作业已经写完了，千纱可以晚上可以好好奖励我么♥?"

    # Create a mocked state
    state: GlobalPersonaState = {
        "timestamp": current_time,

        "user_input": user_input,
        "user_name": "EAMARS",
        "user_profile": await get_user_profile("cc2e831e-2898-4e87-9364-f5d744a058e8"),
        "global_user_id": "cc2e831e-2898-4e87-9364-f5d744a058e8",
        "platform_bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "channel_topic": "日常交流",
        "user_topic": "千纱和EAMARS在房间里聊天",

        "decontexualized_input": user_input,
        "research_facts": f"现在的时间为{current_time}",

        # "character_profile": load_personality("personalities/kazusa.json"),
        "character_profile": await get_character_profile(),

    }

    result = await call_cognition_subgraph(state)
    print(f"Cognition result: {result['action_directives']}")
    
    # for affinity in range(0, 1001, 50):
    #     state["user_affinity_score"] = affinity
    #     result = await call_cognition_subgraph(state)
    #     print(f"Cognition result for affinity {affinity}: {result}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())
