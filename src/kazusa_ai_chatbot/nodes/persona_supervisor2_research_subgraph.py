from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.nodes.persona_supervisor2_schema import GlobalPersonaState
from kazusa_ai_chatbot.agents.web_search_agent2 import web_search_agent
from kazusa_ai_chatbot.agents.memory_retriever_agent import memory_retriever_agent
from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_PERSONA_SUPERVISOR_STAGE1_RETRY

from kazusa_ai_chatbot.utils import parse_llm_json_output, get_llm

import json
import logging
from typing import TypedDict


logger = logging.getLogger(__name__)



class ResearchSubgraphState(TypedDict):
    user_input: str
    should_stop: bool
    retry: int
    timestamp: str

    # dispatcher generated message
    next_action: str
    dispatcher_reasoning: str  # The reason to move to next action
    task: str
    context: dict
    expected_response: str

    # Agent generate message
    research_results: list

    # Evaluator feedback
    evaluator_feedback: str


_RESEARCH_DISPATCHER_PROMPT = """\
你是人格模拟系统的“感知调度员”。你的目标是判断为了让角色做出真实的回应，我们需要检索哪些背景信息。

# 分析逻辑 (Priority)：
1. **反馈至上**：评估员 (Evaluator) 的反馈 (Evaluator feedback) 是你的最高指令。
2. 认知模型处理：
   * 触发条件：大模型通过自身权重即可完美回答。
   * 包括：常识（如：天空颜色、科学定律）、逻辑推理、数学计算、语言翻译、情感安抚、日常寒暄、系统时间查询。
   * "next_action": "evaluator"
3. 内部记忆：
   * 触发条件：涉及用户个人历史、之前的对话约定、角色私有的秘密或特定人际关系。
   * 目的：获取“只有你们俩知道”的信息。
   * "next_action": "memory_retriever_agent"
4. 外部知识：
   * 触发条件：
     a) 具有强时效性（如：今天的新闻、当下的天气、即时股价）。
     b) 极度专业/冷门（如：某个特定API的报错文档、特定经纬度的地图）。
     c) 认知模型知识截止日期之后发生的事件（2024年以后的事实）。
   * "next_action": "web_search_agent"


# 任务指派信息
- "task": "具体要检索的任务描述"
- "context": (可选) 在 context 中提取关键实体（人物、时间、地点）。若不提供则默认为空字典 {}
- "expected_response": 期望的响应格式和内容 (例如: 日期 YYMMDD, 相关来源URL, 是/否)

# 输出要求：
请务必返回合法的 JSON 字符串，包含以下字段：
{
    "next_action": "memory_retriever_agent" | "web_search_agent" | "evaluator",
    "reasoning": "string",
    "task": "string",
    "context": {
        "key": "value",
        ...
    },
    "expected_response": "string"
}
"""
_research_dispatcher_llm = get_llm(temperature=0.5, top_p=1.0)
async def call_research_dispatcher(state: ResearchSubgraphState) -> dict:
    system_prompt = SystemMessage(content=_RESEARCH_DISPATCHER_PROMPT)

    # Build human message
    human_message = HumanMessage(content=state["user_input"])
    evaluator_feedback = state.get("evaluator_feedback", "")
    evaluator_feedback_message = HumanMessage(content=f"Evaluator feedback:\n{evaluator_feedback}", name="evaluator")
    
    # Call LLM
    response = await _research_dispatcher_llm.ainvoke([
        system_prompt, 
        human_message,
        evaluator_feedback_message,
    ])
    
    # Parse response
    try:
        result = parse_llm_json_output(response.content)
        next_action = result.get("next_action")
        task = result.get("task")
        context = result.get("context")
        reasoning = result.get("reasoning")
        expected_response = result.get("expected_response")
    except Exception as e:
        logger.error(f"Failed to parse research dispatcher response: {e}")
        next_action = "evaluator"
        task = ""
        context = {}
        reasoning = f"Failed to parse response, defaulting to evaluator: {e}"
        expected_response = ""

    return {
        "next_action": next_action,
        "dispatcher_reasoning": reasoning,
        "task": task,
        "context": context,
        "expected_response": expected_response
    }

async def call_memory_retriever_agent(state: ResearchSubgraphState) -> dict:
    results = await memory_retriever_agent(
        task=state["user_input"],
        context=state["context"],
        expected_response=state["expected_response"]
    )

    prev_research_results = state.get("research_results", [])

    # Assemble new results
    current_research_result = {
        "fact": results.get("response", ""),
        "metadata": results.get("knowledge_metadata", {})
    }

    return {
        "research_results": prev_research_results + [current_research_result]
    }

async def call_web_search2_agent(state: ResearchSubgraphState) -> dict:
    results = await web_search_agent(
        task=state["user_input"],
        context=state["context"],
        expected_response=state["expected_response"]
    )
    prev_research_results = state.get("research_results", [])
    
    # Assemble new results
    current_research_result = {
        "fact": results.get("response", ""),
        "metadata": results.get("knowledge_metadata", {})
    }

    return {
        "research_results": prev_research_results + [current_research_result]
    }


_RESEARCH_EVALUATOR_PROMPT = """
你是检索质量审计员。对比“任务需求”与“已检索内容”

# 核心准则：
1. **充足性：** 资料是否覆盖了用户问题的核心诉求？
2. **必要性：** 是否存在过度检索？如果认知模型本身能处理（常识/逻辑），应立即停止检索。
3. **真实性：** 检索到的信息是否与角色设定、时间线 {timestamp} 一致？

# 分析逻辑：
- **设置 should_stop: true 的情况：**
  - 检索到的内容已完全覆盖用户需求。
  - 用户问题属于“模型原生知识”（常识、简单计算、日期查询），不需要进一步检索。
  - 重试次数已达上限，必须基于现有信息强行回应。
  - "reasoning": 简述停止继续检索的理由。
- **设置 should_stop: false 的情况：**
  - 检索结果与任务需求不匹配（跑题）。
  - 检索结果存在矛盾，需要另一个工具（如用 memory 验证 web 搜到的内容）进行交叉验证。
  - 关键信息缺失（如：用户问了两个问题，只搜到了一个答案）。
  - "reasoning": 简述理由：已解决哪些问题？缺失哪些关键点？若继续，下一步的优化建议是什么？

# 输入格式
{{
    "user_input": "用户输入",
    "research_results": [已执行的工具、参数及结果摘要],
    "retry": 当前重试次数 n / MAX_RETRY
}}

# 输出要求：
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "should_stop": true | false,
    "reasoning": "string"
}}
"""
_research_evaluator_llm = get_llm(temperature=0.5, top_p=1.0)
async def call_research_evaluator(state: ResearchSubgraphState) -> dict:
    retry = state.get("retry", 0) + 1
    timestamp = state.get("timestamp")

    system_prompt = SystemMessage(content=_RESEARCH_EVALUATOR_PROMPT.format(timestamp=timestamp))

    research_results = state.get("research_results", [])
    result_facts = [r["fact"] for r in research_results]
    
    evaluation_input = {
        "user_input": state["user_input"],
        "research_results": result_facts,
        "retry": f"{retry}/{MAX_PERSONA_SUPERVISOR_STAGE1_RETRY}"
    }

    evaluation_message = HumanMessage(content=json.dumps(evaluation_input))

    # Call LLM
    response = await _research_evaluator_llm.ainvoke([
        system_prompt, 
        evaluation_message,
    ])
    
    # Parse response
    try:
        result = parse_llm_json_output(response.content)
        should_stop = result.get("should_stop")
        reasoning = result.get("reasoning")
    except Exception as e:
        logger.error(f"Failed to parse research evaluator response: {e}")
        should_stop = False
        reasoning = f"Failed to parse response, defaulting to false: {e}"

    # Check if stop
    if retry >= MAX_PERSONA_SUPERVISOR_STAGE1_RETRY:
        should_stop = True

    return {
        "should_stop": should_stop,
        "evaluator_feedback": reasoning
    }

async def call_research_subgraph(state: GlobalPersonaState) -> dict:

    research_builder = StateGraph(ResearchSubgraphState)
    research_builder.add_node("dispatcher", call_research_dispatcher)
    research_builder.add_node("memory_retriever_agent", call_memory_retriever_agent)
    research_builder.add_node("web_search_agent", call_web_search2_agent)
    research_builder.add_node("evaluator", call_research_evaluator)

    # Build edges
    research_builder.add_edge(START, "dispatcher")

    # Condition call
    research_builder.add_conditional_edges(
        "dispatcher",
        lambda state: state["next_action"],
        {
            "memory_retriever_agent": "memory_retriever_agent",
            "web_search_agent": "web_search_agent",
            "evaluator": "evaluator",
        }
    )
    research_builder.add_edge("memory_retriever_agent", "evaluator")
    research_builder.add_edge("web_search_agent", "evaluator")
    
    # Loop
    research_builder.add_conditional_edges(
        "evaluator",
        lambda x: "loop" if not x["should_stop"] else "finish",
        {
            "loop": "dispatcher",
            "finish": END,
        }
    )

    research_subgraph = research_builder.compile()

    # Get attributes
    decontexualized_input = state["decontexualized_input"]

    # initial states
    initial_state = {
        "timestamp": state["timestamp"],
        "user_input": decontexualized_input,
        "context": {},
        "messages": [],
        "should_stop": False,
        "next_action": "dispatcher",
        "retry": 0,
    }

    results = await research_subgraph.ainvoke(initial_state)

    # Post processing
    research_facts = ""

    # If no knowledge is learned then pass the reasoning
    if not results.get("research_results", []):
        research_facts = results.get("evaluator_feedback", "No facts learned")
        research_metadata = []
    else:
        research_facts = "\n".join([r["fact"] for r in results.get("research_results", [])])
        research_metadata = [r["metadata"] for r in results.get("research_results", [])]

    logger.info(
        f"\nDecontexualized input: {decontexualized_input}\n"
        f"  Research facts: {research_facts}\n"
    )
    
    return {
        "research_facts": research_facts,
        "research_metadata": research_metadata,
    }


async def test_main():
    import datetime

    # Create a mocked state
    state: GlobalPersonaState = {
        "decontexualized_input": "千纱是真人么？",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    result = await call_research_subgraph(state)
    print(result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())