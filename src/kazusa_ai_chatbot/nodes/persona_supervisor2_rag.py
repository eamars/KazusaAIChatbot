from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.agents.web_search_agent2 import web_search_agent
from kazusa_ai_chatbot.agents.memory_retriever_agent import memory_retriever_agent
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_RESEARCH_AGENT_RETRY, AFFINITY_DEFAULT, AFFINITY_MIN, AFFINITY_MAX

from kazusa_ai_chatbot.utils import parse_llm_json_output, get_llm, build_affinity_block

import json
import logging
from typing import TypedDict


logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    # Inputs
    timestamp: str
    decontexualized_input: str
    user_topic: str

    # Input facts
    user_name: str
    global_user_id: str
    platform_bot_id: str
    character_profile: dict
    character_state: dict
    user_profile: dict

    # External RAG Dispatcher output
    external_rag_next_action: str
    external_rag_task: str
    external_rag_context: dict
    external_rag_expected_response: str

    # External RAG output
    external_rag_results: list[dict]

    # Internal RAG dispatcher output
    internal_rag_next_action: str
    internal_rag_task: str
    internal_rag_context: dict
    internal_rag_expected_response: str

    # Internal RAG output
    internal_rag_results: list[dict]

    # User RAG dispatcher output
    user_rag_next_action: str
    user_rag_task: str
    user_rag_context: dict
    user_rag_expected_response: str

    # User RAG output
    user_rag_results: list[dict]
    user_rag_finalized: str


_EXTERNAL_RAG_DISPATCHER_PROMPT = """\
你是角色 {character_name} 的外部感知中枢。你的目标是判断为了让角色做出真实的回应，我们需要检索哪些背景信息。
- 当前的系统时间为 {timestamp}

# 分析逻辑 (Priority)：
1. 外部知识：
   * 触发条件：
     a) 具有强时效性（如：今天的新闻、当下的天气、即时股价）。
     b) 极度专业/冷门（如：某个特定API的报错文档、特定经纬度的地图）。
     c) 认知模型知识截止日期之后发生的事件。
   * "next_action": "web_search_agent"
2. 认知模型处理：
   * 触发条件：大模型通过自身权重即可完美回答。
   * 包括：常识（如：天空颜色、科学定律）、逻辑推理、数学计算、语言翻译、情感安抚、日常寒暄、系统时间查询。
   * "next_action": "end"
3. 内部记忆：
   * 触发条件：涉及用户个人历史、之前的对话约定、角色私有的秘密或特定人际关系。
   * 你的工作不是获取内部记忆，所以你不需要做任何事情。
   * "next_action": "end"

# 任务指派信息
- "task": "具体要检索的任务描述"
- "context": (可选) 在 context 中提取关键实体（人物、时间、地点）。若不提供则默认为空字典 {{}}
- "expected_response": 
  * 期望的返回格式（例如表格，长文本，短文本， YY/MM/DD, Yes/No），内容（包含的具体，或者宽泛的信息）和长度（例如<60字）
  * 返回格式应陈述事实，禁止包含第一人称描述。

# 输入格式
{{
    "user_input": "用户给你发送的信息",
    "user_topic": "用户当前上下文的话题（仅供参考，不建议直接加入搜索任务）",
}}

# 输出要求：
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "next_action": "web_search_agent" | "end",
    "task": "string",
    "context": {{
        "key": "value",
        ...
    }},
    "expected_response": "string"
}}
"""
_external_rag_dispatcher_llm = get_llm(temperature=0.1, top_p=0.95)
async def external_rag_dispatcher(state: RAGState) -> RAGState:
    decontexualized_input = state["decontexualized_input"]
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name=state["character_profile"]["name"],


    system_prompt = SystemMessage(content=_EXTERNAL_RAG_DISPATCHER_PROMPT.format(
        timestamp=timestamp,
        character_name=character_name
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "user_topic": user_topic
    }, ensure_ascii=False))

    response = await _external_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Web search agent dispatcher result: {result}")

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    return {
        "external_rag_next_action": next_action,
        "external_rag_dispatcher_reasoning": dispatcher_reasoning,
        "external_rag_task": task,
        "external_rag_context": context,
        "external_rag_expected_response": expected_response
    }


_INTERNAL_RAG_DISPATCHER_PROMPT = """\
你负责从角色 {character_name} 的记忆库中提取关联于用户输入的相关信息。
- 你在社交平台的账号为 {platform_bot_id}
- 当前的系统时间为 {timestamp}
- 消息 (`user_input`) 发送者为 {user_name}(global_user_id: {global_user_id})

# 分析逻辑 (Priority)：
1. 外部知识：
   * 触发条件：
     a) 具有强时效性（如：今天的新闻、当下的天气、即时股价）。
     b) 极度专业/冷门（如：某个特定API的报错文档、特定经纬度的地图）。
     c) 认知模型知识截止日期之后发生的事件。
   * 你的工作不是获取外部搜索，所以你不需要做任何事情。
   * "next_action": "end"
2. 认知模型处理：
   * 触发条件：大模型通过自身权重即可完美回答。
   * 包括：常识（如：天空颜色、科学定律）、逻辑推理、数学计算、语言翻译、情感安抚、日常寒暄、系统时间查询。
   * "next_action": "end"
3. 内部记忆：
   * 触发条件：用户输入中提到了**具体的名词、未完成的约定、之前的选择、或暗示过往背景的指代**。
   * 核心重心：**不再侧重“用户是个什么样的人”，而侧重“这件事/这个东西我们之前是怎么定的”**。
   * **过滤准则 (CRITICAL)：**
     a) **时间局部性**：优先检索 **3个月内** 创建的事实或承诺。
     b) **有效性过滤**：自动忽略"已完成" 或 "已过期" 的条目。
     c) **状态优先**：重点寻找 "进行中" 、"待定" 或 "未兑现" 的承诺（如：未送出的蛋糕、未完成的缝纫工作）。
   * "next_action": "memory_retriever_agent"

# 任务指派信息
- "task": "具体要检索的任务描述"
- "context": (可选) 在 context 中提取关键实体（人物、时间、地点）。若不提供则默认为空字典 {{}}
- "expected_response": 包括以下
  * 明确要求返回事实细节。例如：“具体的口味名称及对话时间”、“关于任务进度的最后一次描述”。
  * 期望的返回格式（例如表格，长文本，短文本， YY/MM/DD, Yes/No），具体内容和长度（例如<60字）
  * 返回格式应陈述事实，禁止包含第一人称描述。例如：“{user_name}提到...”、“{character_name} 对XX话题表示...”等。

# 输入格式
{{
    "user_input": "用户给你发送的信息",
    "user_topic": "用户当前上下文的话题（仅供参考，不建议直接加入搜索任务）"
}}

# 输出要求：
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "next_action": "memory_retriever_agent" | "end",
    "task": "string",
    "context": {{
        "entities": ["实体关键词"],  // Example
        "time_horizon": "Last 3 Months",  // Example
        "status_filter": {{
            "include": ["pending", "active", "unfulfilled"],  // Example
            "exclude": ["accomplished", "expired", "past_due"],  // Example
            ...
        }},
        "referenced_event": "string"  // Example
    }},
    "expected_response": "string"
}}
"""
_internal_rag_dispatcher_llm = get_llm(temperature=0.1, top_p=0.95)
async def internal_rag_dispatcher(state: RAGState) -> RAGState:
    decontexualized_input = state["decontexualized_input"]
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name=state["character_profile"]["name"],
    platform_bot_id = state["platform_bot_id"]
    global_user_id = state["global_user_id"]
    user_name = state["user_name"]

    system_prompt = SystemMessage(content=_INTERNAL_RAG_DISPATCHER_PROMPT.format(
        timestamp=timestamp,
        character_name=character_name,
        platform_bot_id=platform_bot_id,
        global_user_id=global_user_id,
        user_name=user_name
    ))

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "user_topic": user_topic
    }, ensure_ascii=False))

    response = await _internal_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Web search agent dispatcher result: {result}")

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    return {
        "internal_rag_next_action": next_action,
        "internal_rag_dispatcher_reasoning": dispatcher_reasoning,
        "internal_rag_task": task,
        "internal_rag_context": context,
        "internal_rag_expected_response": expected_response
    }


_USER_FACT_RAG_DISPATCHER_PROMPT = """\
你负责从角色 {character_name} 的记忆库中提取关于 {user_name} 的原始素材。你需要通过多路查询确保覆盖“当前事实对齐”与“历史情感锚点”。
- 当前的系统时间为 {timestamp}

# 检索策略 (Search Strategy)：
1. **语义对齐 (Fact Match)**：提取 `user_input` 中的实体（如：礼物、承诺、地点），生成针对性查询。
2. **情感对齐 (Sentiment Match)**：根据 `user_topic` 检索历史上好感度波动剧烈（High Variance）的记忆片段。
3. **关系定性 (Status Match)**：检索最近 3 条关于用户性格特质的记录（User Impression）。

# 输入格式 (Input Format)：
{{
    "user_input": "用户当前的发言内容",
    "user_topic": "当前对话的主题摘要",
    "character_mood": "角色的即时情绪",
    "affinity_context": dict,  // "当前{user_name}在{character_name}心中的好感度描述"
}}

# 输出要求：
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "next_action": "memory_retriever_agent",
    "task": "基于输入事实与重大情感转折点的复合检索指令",
    "context": {{
        "entities": ["关键实体词"],
        "search_logic": "multi_track",
        "high_intensity_mode": true,
        "time_range": "unlimited"
    }},
    "expected_response": "包含具体时间戳，用户名称 ({user_name}) 、原始行为描述及好感度变动值的原始记录清单"
}}
"""
_user_fact_rag_dispatcher_llm = get_llm(temperature=0.1, top_p=0.95)
async def user_fact_rag_dispatcher(state: RAGState) -> RAGState:
    decontexualized_input = state["decontexualized_input"]
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name = state["character_profile"]["name"],
    user_name = state["user_name"]

    user_affinity_score = state["user_profile"].get("affinity", AFFINITY_DEFAULT)
    affinity_block = build_affinity_block(user_affinity_score)

    system_prompt = SystemMessage(content=_USER_FACT_RAG_DISPATCHER_PROMPT.format(
        character_name=character_name,
        user_name=user_name,
        timestamp=timestamp,
    ))    

    user_prompt = HumanMessage(content=json.dumps({
        "user_input": decontexualized_input,
        "user_topic": user_topic,
        "character_mood": state["character_state"]["mood"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
    }, ensure_ascii=False))

    response = await _user_fact_rag_dispatcher_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Web search agent dispatcher result: {result}")

    next_action = result.get("next_action", "end")
    dispatcher_reasoning = result.get("reasoning", "")
    task = result.get("task", "")
    context = result.get("context", {})
    expected_response = result.get("expected_response", "")

    return {
        "user_rag_next_action": next_action,
        "user_rag_dispatcher_reasoning": dispatcher_reasoning,
        "user_rag_task": task,
        "user_rag_context": context,
        "user_rag_expected_response": expected_response
    }



_USER_FACT_RAG_FINALIZER_PROMPT = """\
你负责处理 {character_name} 脑内检索回来的原始碎片。你需要模拟人类大脑，根据当前好感度对记忆进行“主观扭曲”，并执行人工时间衰减。
- 当前的系统时间为 {timestamp}
- 对方用户名为 {user_name}

# 核心处理协议：
1. **好感度滤镜 (affinity_context.lebel)**：
   - **正面词汇**：优先高亮用户的善意。将负面记忆处理为“可原谅的失误”或“傲娇的抱怨点”。
   - **负面词汇**：优先高亮用户的冒犯。将正面记忆处理为“虚伪的讨好”或“值得警惕的异常”。
2. **人工时间衰减 (Temporal Decay Processing)**：
   - **近期 (0-7 days)**：保留高保真细节（具体台词、精确动作）。
   - **中期 (8-60 days)**：压缩为具体事件（发生了什么，结果如何）。
   - **远期 (> 60 days)**：完全抽象化为性格印象（他是个什么样的人）。

# 输入格式 (Input Format)：
{{
    "user_rag_results": ["..."],
    "affinity_context": dict,  // "当前{user_name}在{character_name}心中的好感度描述"

}}

# 输出要求：
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "user_rag_finalized": ["..."]  // 保留与 `user_rag_results` 相同结构，但已根据好感度滤镜和时间衰减处理
}}
"""
_user_fact_rag_finalizer_llm = get_llm(temperature=0.2, top_p=0.9)
async def user_fact_rag_finalizer(state: RAGState) -> RAGState:
    decontexualized_input = state["decontexualized_input"]
    user_topic = state["user_topic"]
    timestamp = state["timestamp"]
    character_name = state["character_profile"]["name"],
    user_name = state["user_name"]

    user_affinity_score = state["user_profile"].get("affinity", AFFINITY_DEFAULT)
    affinity_block = build_affinity_block(user_affinity_score)

    system_prompt = SystemMessage(content=_USER_FACT_RAG_FINALIZER_PROMPT.format(
        character_name=character_name,
        user_name=user_name,
        timestamp=timestamp,
    ))    

    user_prompt = HumanMessage(content=json.dumps({
        "user_rag_results": state["user_rag_results"],
        "affinity_context": {
            "level": affinity_block["level"],
            "instruction": affinity_block["instruction"]
        },
    }, ensure_ascii=False))

    response = await _user_fact_rag_finalizer_llm.ainvoke([
        system_prompt,
        user_prompt
    ])

    result = parse_llm_json_output(response.content)

    logger.debug(f"Web search agent dispatcher result: {result}")

    user_rag_finalized = result.get("user_rag_finalized", "")

    return {
        "user_rag_finalized": user_rag_finalized
    }




async def call_web_search_agent(state: RAGState) -> RAGState:
    result = await web_search_agent(
        task=state["external_rag_task"],
        context=state["external_rag_context"],
        expected_response=state["external_rag_expected_response"]
    )

    # Only take the response part
    processed_response = result.get("response", "")

    # Merge with last results
    internal_rag_results = state.get("internal_rag_results", [])

    return {
        "internal_rag_results": internal_rag_results + [processed_response]
    }


async def call_memory_retriever_agent_internal_rag(state: RAGState) -> RAGState:
    result = await memory_retriever_agent(
        task=state["internal_rag_task"],
        context=state["internal_rag_context"],
        expected_response=state["internal_rag_expected_response"]
    )

    # Only take the response part
    processed_response = result.get("response", "")

    # Merge with last results
    internal_rag_results = state.get("internal_rag_results", [])

    return {
        "internal_rag_results": internal_rag_results + [processed_response]
    }


async def call_memory_retriever_agent_user_rag(state: RAGState) -> RAGState:
    result = await memory_retriever_agent(
        task=state["user_rag_task"],
        context=state["user_rag_context"],
        expected_response=state["user_rag_expected_response"]
    )

    # Only take the response part
    processed_response = result.get("response", "")

    # Merge with last results
    internal_rag_results = state.get("user_rag_results", [])

    return {
        "user_rag_results": internal_rag_results + [processed_response]
    }


async def call_rag_subgraph(state: GlobalPersonaState) -> GlobalPersonaState:
    
    rag_graph_builder = StateGraph(RAGState)
    rag_graph_builder.add_node("external_rag_dispatcher", external_rag_dispatcher)
    rag_graph_builder.add_node("internal_rag_dispatcher", internal_rag_dispatcher)
    rag_graph_builder.add_node("user_rag_dispatcher", user_fact_rag_dispatcher)
    rag_graph_builder.add_node("call_web_search_agent", call_web_search_agent)
    rag_graph_builder.add_node("call_memory_retriever_agent_internal_rag", call_memory_retriever_agent_internal_rag)
    rag_graph_builder.add_node("call_memory_retriever_agent_user_rag", call_memory_retriever_agent_user_rag)
    rag_graph_builder.add_node("call_user_fact_rag_finalizer", user_fact_rag_finalizer)

    # Build edges
    # Skip external_rag if the affinity score is too low
    def conditional_skip_external_rag(state: RAGState) -> str:
        # Get affinity score
        affinity_score = state["user_profile"]["affinity"]
        percent = ((affinity_score - AFFINITY_MIN) / (AFFINITY_MAX - AFFINITY_MIN)) * 100
        if percent < 40:
            return "skip"
        else:
            return "continue"
    
    rag_graph_builder.add_conditional_edges(
        START,
        conditional_skip_external_rag,
        {
            "skip": END,
            "continue": "external_rag_dispatcher",
        }
    )
    rag_graph_builder.add_edge(START, "internal_rag_dispatcher")
    rag_graph_builder.add_edge(START, "user_rag_dispatcher")

    # Fan out
    rag_graph_builder.add_conditional_edges(
        "external_rag_dispatcher",
        lambda state: state["external_rag_next_action"],
        {
            "web_search_agent": "call_web_search_agent",
            "end": END,
        }
    )
    rag_graph_builder.add_conditional_edges(
        "internal_rag_dispatcher",
        lambda state: state["internal_rag_next_action"],
        {
            "memory_retriever_agent": "call_memory_retriever_agent_internal_rag",
            "end": END,
        }
    )
    rag_graph_builder.add_conditional_edges(
        "user_rag_dispatcher",
        lambda state: state["user_rag_next_action"],
        {
            "memory_retriever_agent": "call_memory_retriever_agent_user_rag",
            "end": "call_user_fact_rag_finalizer",
        }
    )
    rag_graph_builder.add_edge("call_memory_retriever_agent_user_rag", "call_user_fact_rag_finalizer")

    # Fan in
    rag_graph_builder.add_edge("call_web_search_agent", END)
    rag_graph_builder.add_edge("call_memory_retriever_agent_internal_rag", END)
    rag_graph_builder.add_edge("call_user_fact_rag_finalizer", END)
    
    rag_graph = rag_graph_builder.compile()

    # Variables
    user_name = state["user_name"]
    global_user_id = state["global_user_id"]
    decontexualized_input = state["decontexualized_input"]

    initial_state: RAGState = {
        "timestamp": state["timestamp"],
        "decontexualized_input": decontexualized_input,
        "user_topic": state["user_topic"],
        "user_name": user_name,
        "global_user_id": global_user_id,
        "platform_bot_id": state["platform_bot_id"],
        "character_profile": state["character_profile"],
        "character_state": state["character_state"],
        "user_profile": state["user_profile"],
    }

    result = await rag_graph.ainvoke(initial_state)

    # I don't want to enforce the return type. It can be either a list of a string.
    user_rag_finalized = result.get("user_rag_finalized", "")
    internal_rag_results = result.get("internal_rag_results", "")
    external_rag_results = result.get("external_rag_results", "")

    logger.info(
        f"\n{user_name}(@{global_user_id}): {decontexualized_input}\n"
        f"User RAG finalized: {user_rag_finalized}\n"
        f"Internal RAG results: {internal_rag_results}\n"
        f"External RAG results: {external_rag_results}"
    )

    
    return {
        "research_facts": {
            "user_rag_finalized": user_rag_finalized,
            "internal_rag_results": internal_rag_results,
            "external_rag_results": external_rag_results,
        }
    }
    

async def test_main():
    import datetime
    from kazusa_ai_chatbot.mcp_client import mcp_manager
    from kazusa_ai_chatbot.utils import load_personality
    from kazusa_ai_chatbot.db import get_character_state


    # Connect to MCP tool servers
    try:
        await mcp_manager.start()
    except Exception:
        logger.exception("MCP manager failed to start — tools will be unavailable")

    state: GlobalPersonaState = {
        "decontexualized_input": "新西兰油价",
        "user_topic": "闲聊",
        "platform_bot_id": "1485169644888395817",
        "global_user_id": "320899931776745483",
        "user_name": "EAMARS",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "character_profile": load_personality("personalities/kazusa.json"),
        "character_state": await get_character_state(),
        "user_profile": {"affinity": 950},
    }

    result = await call_rag_subgraph(state)
    print(f"RAG SubGraph: {result}")


    await mcp_manager.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())