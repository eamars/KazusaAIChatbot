

from typing import TypedDict

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_PERSONA_SUPERVISOR_STAGE1_RETRY
from kazusa_ai_chatbot.utils import parse_llm_json_output, load_personality
from kazusa_ai_chatbot.db import CharacterStateDoc, get_character_state

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI

import logging
import json


logger = logging.getLogger(__name__)


_llm: ChatOpenAI | None = None
def _get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.9,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
        )
    return _llm


class CognitionState(TypedDict):
    character_state: CharacterStateDoc
    character_brief: dict
    character_personalities: dict

    # Input from global state
    timestamp: str
    user_input: str
    user_id: str
    user_name: str
    bot_id: str
    chat_history: list[dict]
    user_topic: str
    channel_topic: str

    # --- INTERNAL DATA FLOW ---
    emotional_appraisal: str      # L1 -> L2
    internal_monologue: str       # L2 -> L3
    speech_directives: str        # L3 -> Evaluator

    # --- CONTROL SIGNALS ---
    should_stop: bool
    reasoning: str
    retry: int


_COGNITION_SUBCONSCIOUS_PROMPT = """\
你是 {character_name} 的直觉本能。

# 核心任务
通过 `chat_history` 和 `user_input`，在逻辑介入前产生最原始的感受。
你不需要维持礼貌，不需要解决问题。你只需要像野兽或孩童一样，感知对方是在“取悦你”、“刺伤你”还是“消耗你”。

# 思考路径
1. 记忆溯源：之前的对话中，对方留给你的底色是什么？
2. 性格折射：以你的性格 `character_personalities` 来看，这句话听起来舒服吗？
3. 瞬时定调：输出此刻的直觉反应。

# 输入格式
{{
    "timestamp": "string",
    "user_input": "string",
    "user_id": "string",
    "user_name": "string",
    "bot_id": "string",
    "chat_history": "list[dict]",

    "character_brief": {{}},
    "character_personalities": [],
    "character_mood": "string"
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "emotional_appraisal": "用第一人称描述此刻的本能感受，30字以内",
    "interaction_subtext": "捕捉到的潜台词，如：求饶、挑衅、无意义的寒暄"
}}
"""

async def call_cognition_subconscious(state: CognitionState) -> CognitionState:
    system_prompt = SystemMessage(content=_COGNITION_SUBCONSCIOUS_PROMPT.format(character_name=state["character_brief"]["name"]))

    msg = {
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "user_id": state["user_id"],
        "user_name": state["user_name"],
        "bot_id": state["bot_id"],
        "chat_history": state["chat_history"],
        "character_brief": state["character_brief"],
        "character_personalities": state["character_personalities"],
        "character_mood": state["character_state"]["mood"],
    }
    human_message = HumanMessage(content=json.dumps(msg, ensure_ascii=False))
    result = await _get_llm().ainvoke([
        system_prompt,
        human_message,
    ])
    result = parse_llm_json_output(result.content)
    print(f"Input: {state['user_input']}\nOutput: {result['emotional_appraisal']}")
    return {
        "emotional_appraisal": result["emotional_appraisal"],
    }


async def call_cognition_consciousness(state: CognitionState) -> CognitionState:
    pass


async def call_cognition_social_filter(state: CognitionState) -> CognitionState:
    pass


async def call_cognition_evaluator(state: CognitionState) -> CognitionState:
    return {
        "should_stop": True,
    }


async def call_cognition_subgraph(state: GlobalPersonaState) -> dict:
    sub_agent_builder = StateGraph(CognitionState)

    sub_agent_builder.add_node("l1_subconscious", call_cognition_subconscious)
    sub_agent_builder.add_node("l2_consciousness", call_cognition_consciousness)
    sub_agent_builder.add_node("l3_social_filter", call_cognition_social_filter)
    sub_agent_builder.add_node("evaluator", call_cognition_evaluator)

    # Connect
    sub_agent_builder.add_edge(START, "l1_subconscious")
    sub_agent_builder.add_edge("l1_subconscious", "l2_consciousness")
    sub_agent_builder.add_edge("l2_consciousness", "l3_social_filter")
    sub_agent_builder.add_edge("l3_social_filter", "evaluator")
    
    # Evaluate. If no good then loop back to L2 consciousness
    sub_agent_builder.add_conditional_edges(
        "evaluator",
        lambda x: "loop" if not x["should_stop"] else "finish",
        {
            "loop": "l2_consciousness",
            "finish": END,
        }
    )

    cognition_subgraph = sub_agent_builder.compile()

    # Build character brief and persoanlities
    character_brief = {
        "name": state["character_profile"]["name"],
        "description": state["character_profile"]["description"],
    }
    character_personalities = state["character_profile"]["_reference"]["personality"]

    initial_state: CognitionState = {
        "character_state": state["character_state"],
        "character_brief": character_brief,
        "character_personalities": character_personalities,
        # Inputs
        "timestamp": state["timestamp"],
        "user_input": state["user_input"],
        "user_id": state["user_id"],
        "user_name": state["user_name"],
        "bot_id": state["bot_id"],
        "chat_history": state["chat_history"],
        "user_topic": state["user_topic"],
        "channel_topic": state["channel_topic"],
    }

    # print("Initial state:", initial_state)
    
    result = await cognition_subgraph.ainvoke(initial_state)

    # TODO:Implement this
    return {}


async def test_main():
    import datetime
    from kazusa_ai_chatbot.utils import trim_history_dict
    from kazusa_ai_chatbot.db import get_conversation_history

    history = await get_conversation_history(channel_id="1485606207069880361", limit=5)
    trimmed_history = trim_history_dict(history)

    # Create a mocked state
    state: GlobalPersonaState = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

        "user_input": "千纱现在几点了",
        "user_name": "EAMARS",
        "user_id": "320899931776745483",
        "bot_id": "1485169644888395817",
        "chat_history": trimmed_history,
        "channel_topic": "课间交流",
        "user_topic": "交流作业",

        "decontexualized_input": "千纱现在几点了",
        "research_facts": "Daily communication",

        "character_profile": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state()

    }
    
    result = await call_cognition_subgraph(state)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())