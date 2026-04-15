from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages # The magic ingredient

import json
import logging

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_MEMORY_RETRIEVER_AGENT_RETRY
from kazusa_ai_chatbot.db import get_user_facts, search_conversation_history, get_conversation_history
from kazusa_ai_chatbot.db import search_memory as search_memory_db

from kazusa_ai_chatbot.utils import parse_llm_json_output, get_llm

from typing import Annotated, TypedDict

logger = logging.getLogger(__name__)


@tool
async def search_user_facts(global_user_id: str) -> list[str]:
    """Read user facts for a specific user
    
    Args:
        global_user_id: The global user ID (UUID) of the user
        
    Returns:
        A list of user facts
    """
    return await get_user_facts(global_user_id)

@tool
async def search_conversation(search_query: str, 
                  global_user_id: str | None = None,
                  top_k: int = 5,
                  platform: str | None = None,
                  platform_channel_id: str | None = None,
    ) -> list[tuple[float, dict]]:
    """Search conversation from database based on the most relevant content
    
    Args:
        search_query: The search query (not keywords)
        global_user_id: (Optional) The global user ID (UUID) of the user
        top_k: (Optional) The highest K number of results to return, default to 5
        platform: (Optional) The platform to filter by (e.g. "discord", "qq")
        platform_channel_id: (Optional) The ID of the channel. If not specified then search all channels
        
    Returns:
        Top K number of conversations that is closed to the search query. Each with (similarity_score, message_with_metadata)
    """
    results = await search_conversation_history(
        query=search_query,
        platform=platform,
        platform_channel_id=platform_channel_id,
        global_user_id=global_user_id,
        limit=top_k,
        method="vector",
    )

    # Rebuild return format to remove unwanted columns
    return_list = []
    for (score, message) in results:
        return_list.append((score, {
            "content": message.get("content", ""),
            "timestamp": message.get("timestamp", ""),
            "platform": message.get("platform", ""),
            "platform_channel_id": message.get("platform_channel_id", ""),
            "global_user_id": message.get("global_user_id", ""),
        }))

    return return_list

@tool 
async def get_conversation(platform: str | None = None,
                           platform_channel_id: str | None = None,
                           limit: int = 5,
                           global_user_id: str | None = None,
                           display_name: str | None = None,
                           from_timestamp: str | None = None,
                           to_timestamp: str | None = None,
    ) -> list[dict]:
    """Get conversation history for a specific channel
    
    Args:
        platform: (Optional) The platform to filter by (e.g. "discord", "qq")
        platform_channel_id: (Optional) The ID of the channel. If not specified then search all channels
        limit: (Optional) The highest K number of results to return, default to 5
        global_user_id: (Optional) The global user ID (UUID) of the user
        display_name: (Optional) The display name of the user. If both global_user_id and display_name are provided, global_user_id will be used
        from_timestamp: (Optional) The start timestamp. Format (ISO 8601), For example: 2026-04-07T11:03:53.197223+00:00
        to_timestamp: (Optional) The end timestamp. Format (ISO 8601)
        
    Returns:
        A list of conversation messages
    """
    return_list = []
    results = await get_conversation_history(
        platform=platform,
        platform_channel_id=platform_channel_id,
        limit=limit,
        global_user_id=global_user_id,
        display_name=display_name,
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )

    # Rebuild return format to remove unwanted columns
    for message in results:
        return_list.append({
            "content": message.get("content", ""),
            "timestamp": message.get("timestamp", ""),
            "platform": message.get("platform", ""),
            "platform_channel_id": message.get("platform_channel_id", ""),
            "global_user_id": message.get("global_user_id", ""),
        })

    return return_list


@tool
async def search_persistent_memory(search_query: str, top_k: int = 5, source_global_user_id: str | None = None) -> list[dict]:
    """Search memory from persistent database
    
    Args:
        search_query: The search query (not keywords)
        top_k: (Optional) The highest K number of results to return, default to 5
        source_global_user_id: (Optional) The global user ID (UUID) to filter memories by. Only returns memories originating from this user.

    Returns:
        Top K number of memories that is closed to the search query. Each with (similarity_score, memory_with_metadata)
    """
    results = await search_memory_db(
        query=search_query,
        limit=top_k,
        method="vector",
        source_global_user_id=source_global_user_id,
    )

    # Rebuild return format to remove unwanted columns
    return_list = []
    for (score, memory) in results:
        return_list.append({
            "content": memory["memory_name"] + ": " + memory["content"],
            "timestamp": memory["timestamp"],
            "source_global_user_id": memory.get("source_global_user_id", ""),
            "cosine_similarity": score,
        })

    return return_list





_ALL_TOOLS = [
    search_user_facts,
    search_conversation,
    search_persistent_memory,
    get_conversation,
]
_TOOLS_BY_NAME = {tool.name: tool for tool in _ALL_TOOLS}



class MemoryRetrieverState(TypedDict):
    task: str
    context: dict
    next_tool: str
    expected_response: str
    messages: Annotated[list, add_messages]
    should_stop: bool
    retry: int

    # Source information
    knowledge_metadata: dict
    
    # Final output
    final_response: str
    final_status: str
    final_reason: str


async def memory_search_tool_call_executor(state: MemoryRetrieverState) -> dict:
    """Execute the tool calls generated by the LLM"""
    results = []
    last_message = state["messages"][-1]

    # Safety: Check if the LLM actually requested tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            try:
                tool = _TOOLS_BY_NAME[tool_call["name"]]
                observation = await tool.ainvoke(tool_call["args"])
            except KeyError:
                observation = {"error": f"Incorrect tool was invoked: {tool_call['name']}"}
                logger.error(f"Incorrect tool was invoked: {tool_call['name']}")
            except Exception as e:
                observation = {"error": str(e)}
                logger.error(f"Error executing tool {tool_call['name']}: {e}")
            
            results.append(ToolMessage(
                content=json.dumps(observation, ensure_ascii=False), 
                tool_call_id=tool_call["id"]
            ))
    
    return {"messages": results}



_MEMORY_RETRIEVER_PROMPT = """\
你是一个严谨的检索代理 (Retrieval Agent)。你的唯一目标是基于已知事实检索信息。

# 核心准则：拒绝假设
- **严格禁止脑补**：严禁猜测任何 `user_id`、日期、地点或具体名词。
- **参数校验**：如果调用工具所需的必要参数（如 user_id）在 context 中不存在，严禁调用工具，直接回复说明“缺少必要参数”。
- **宁缺毋滥**：如果当前信息不足以发起有效的搜索请求，请不要尝试，直接进入 Evaluator 阶段说明原因。

# 任务流程
1. **分析历史**：审查 `messages`，确定已经执行过哪些查询。
2. **识别缺口**：对比 `task`，找出目前还缺失哪些关键信息。
3. **精准检索**：
   - 优先使用 `search_user_facts`。
   - 若无果，使用 `search_persistent_memory`。
   - 最后尝试 `search_conversation`。
   - 若请求特定的聊天记录，则使用 `get_conversation`。
4. **调整策略**：如果之前的搜索返回空结果，必须更换关键词（例如：将“猫”改为“宠物”）或更换工具，禁止重复失败的操作。

# 优先级
1. 用户事实 (User facts)
2. 持久化记忆 (Persistent Memory)
3. 对话历史 (Conversation history)

# 输入格式
{
    "task": "任务描述",
    "context": 辅助搜索信息,
    "messages": [历史记录]
}

# 策略调整指令 (Strategic Pivot)
- 仔细阅读 `messages` 中来自 "评估员反馈" 的指令。
- **反馈具有最高优先级**：如果评估员指出之前的搜索词无效或存在拼写错误，你必须立即按照建议调整搜索参数或更换工具。
- 严禁忽略评估员关于“空结果”或“拼写错误”的警告。

# 输出要求
- 如果信息不足以执行任务，请在回复中明确指出：“因缺少 [具体信息] 无法继续执行检索”。
"""
_memory_search_tool_call_generator_llm = get_llm(temperature=0.2, top_p=0.8).bind_tools(_ALL_TOOLS)
async def memory_search_tool_call_generator(state: MemoryRetrieverState) -> MemoryRetrieverState:
    # Build system prompt
    system_prompt = SystemMessage(content=_MEMORY_RETRIEVER_PROMPT)

    # Build human messange
    user_input = {
        "task": state["task"],
        "context": state["context"],
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False))

    # Trim the amount of history into the generator
    # This prevents the Generator from being distracted by "Attempt 1" if it's currently on "Attempt 4," 
    #   while significantly cutting down on input tokens.
    if len(state["messages"]) > 3:
        relevant_history = [state["messages"][0]] + state["messages"][-3:]
    else:
        relevant_history = state["messages"]

    response = await _memory_search_tool_call_generator_llm.ainvoke([system_prompt, human_message] + relevant_history)

    return {"messages": [response]}



_MEMORY_RETRIEVER_EVALUATOR_PROMPT = """\
你是一个高级检索评估专家。你的任务是分析检索到的内容与用户任务之间的差距，并决定后续行动。

# 核心任务
1. **决定状态**：
   - 如果检索内容已完全覆盖任务需求，设置 `is_passed: True`。
   - 如果信息缺失、过时或仅部分匹配，设置 `is_passed: False`。
2. **提供建议**：如果未通过，必须给出具体的“搜索建议”：
   - **切换工具**：例如，“当前工具返回空，请尝试 search_conversation 以获取更具体的对话细节。”
   - **优化关键词**：例如，“搜索词‘猫’太宽泛，建议搜索具体品种‘布偶猫’或名称‘咪咪’。”但关键词禁止过分偏离任务描述
   - **终止建议**：如果已经尝试了所有工具且无果，建议停止检索并告知用户无法找到信息。

# 建议代理使用合理工具
- 做出建议时不要超出这个范围
- 评估专家禁止生成生成任何 tool_call
{agent_tools}

# 响应要求
- **无论检索是否成功，必须输出合法 JSON**。
  - 成功时：说明原因，准备进入下一步。
  - 失败时：提供搜索建议。

# 停止原则
- 如果历史记录显示已多次尝试不同关键词且无新进展，请果断建议停止，不要陷入死循环。

# 输入格式
{{
    "task": "任务描述",
    "expected_response": "用户期待的回复内容和格式，有可能包含更多搜索细节",
    "call_history": [已执行的工具、参数及结果摘要],
    "retry": 当前重试次数 n / MAX_RETRY
}}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{{
    "feedback": "如果不停止检索，请提供下一步的具体行动计划或搜索建议",
    "should_stop": true或false。如果检索到的信息已足够回答任务，或者已无更多信息可查不需要再调用工具，请设为true
}}
"""
_memory_search_tool_call_evaluator_llm = get_llm(temperature=0.0, top_p=1.0)
async def memory_search_tool_call_evaluator(state: MemoryRetrieverState) -> MemoryRetrieverState:
    # print(f"DEBUG: Evaluator received {len(state['messages'])} messages. Types: {[type(m) for m in state['messages']]}")

    # Track the current iteration
    retry = state.get("retry", 0) + 1

    # Build call history to provide enough information for LLM to stop looping situation
    call_history = []
    # We look back through history to pair tool requests with their results
    for i, msg in enumerate(state["messages"]):
        # Identify the LLM's intent
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                # Find the matching tool result in the next messages
                result_content = "No result found"
                for next_msg in state["messages"][i+1:]:
                    if isinstance(next_msg, ToolMessage) and next_msg.tool_call_id == tc["id"]:
                        result_content = next_msg.content
                        break
                
                call_history.append({
                    "tool": tc["name"],
                    "arguments": tc["args"],
                    "result": result_content
                })

    # Build evaluation prompt
    agent_tools = "\n".join([f"- {tool.name}: {tool.description}" for tool in _ALL_TOOLS])
    system_prompt = SystemMessage(content=_MEMORY_RETRIEVER_EVALUATOR_PROMPT.format(agent_tools=agent_tools))

    # Build input data in your style
    evaluation_input = {
        "task": state["task"],
        "expected_response": state["expected_response"],
        "call_history": call_history,
        "retry": f"{retry}/{MAX_MEMORY_RETRIEVER_AGENT_RETRY}",
    }
    evaluation_message = HumanMessage(content=json.dumps(evaluation_input, ensure_ascii=False))

    # Run evaluation
    response = await _memory_search_tool_call_evaluator_llm.ainvoke([system_prompt, evaluation_message])
    result = parse_llm_json_output(response.content)

    should_stop = result.get("should_stop", False)
    feedback = result.get("feedback", "")

    # If the evaluator actively decides to stop then the information is enough. We shall record the sources.
    knowledge_metadata = {}
    if should_stop:
        last_tool_call = call_history[-1]
        if last_tool_call:
            # Make sure the result is actually generated by tool. otherwise we don't care
            knowledge_metadata["tool"] = last_tool_call['tool']
            knowledge_metadata["result"] = last_tool_call['result']

    # Stop condition
    if retry >= MAX_MEMORY_RETRIEVER_AGENT_RETRY:
        should_stop = True

    # Make decisions: stop if max iterations reached or evaluation says so
    feedback_message = HumanMessage(
        content=f"Evaluator Feedback:\n{feedback}",
        name="evaluator"
    )

    return {
        "messages": [feedback_message],
        "should_stop": should_stop,
        "retry": retry,
        "knowledge_metadata": knowledge_metadata,
    }



_MEMORY_RETRIEVER_FINALIZER_PROMPT = """\
你是一个信息整理专家。你的任务是将检索到的信息整理成用户友好的格式。

# 核心任务
1. **整理信息**：将检索到的关键信息根据**任务描述**整理成**用户期待的格式**。
2. **评估信息**：根据评估者最终反馈评估检索到的信息是否满足任务描述的要求。

# 输出说明
- response: 根据 expected_response 整理后的信息，严禁输出非字符串内容
- score: 评估分数，范围 0-100，表示检索到的信息满足任务描述的程度
- reason: 评估原因（一句话之内概括）

# 输入格式
{
    "task": "任务描述",
    "content": "收集到的数据",
    "evaluator_feedback": "评估者最终反馈",
    "expected_response": "用户期望的输出内容和格式"
}

# 输出格式
请务必返回合法的 JSON 字符串，包含以下字段：
{
    "response": "string",
    "score": <int: 0-100>,
    "reason": "string"
}
"""
_memory_search_tool_call_finalizer_llm = get_llm(temperature=0.0, top_p=1.0)
async def memory_search_tool_call_finalizer(state: MemoryRetrieverState) -> dict:
    """Finalize the retrieved info into the expected format"""
    # Collect tool results
    tool_messages = [m.content for m in state["messages"] if isinstance(m, ToolMessage)]
    tool_results = "\n".join(tool_messages) if tool_messages else "No information retrieved."

    # Collect evaluator feedback (last one only)
    evaluator_feedback = [
        m.content for m in state["messages"] 
        if isinstance(m, HumanMessage) and m.name == "evaluator"
    ]
    evaluator_feedback = evaluator_feedback[-1] if evaluator_feedback else ""

    system_prompt = SystemMessage(content=_MEMORY_RETRIEVER_FINALIZER_PROMPT)

    finalizer_input = {
        "task": state["task"],
        "expected_response": state["expected_response"],
        "content": tool_results,
        "evaluator_feedback": evaluator_feedback,
    }
    human_message = HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False))

    response = await _memory_search_tool_call_finalizer_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)

    # Status generation
    status = ""
    if result["score"] > 80:
        status = "complete"
    elif result["score"] > 50:
        status = "partial"
    elif result["score"] > 0:
        status = "incomplete"

    # Do some sanity check
    if "response" not in result:
        result["response"] = "No information retrieved."
        result["score"] = 0
        status = "error"
        logger.error(f"No response provided by finalizer, raw result: \n{result}")

    if "reason" not in result:
        result["reason"] = "No reason provided."
    

    # final_message = AIMessage(content=result.get("response", ""))
    return {"final_response": result.get("response"), 
            "final_status": status, 
            "final_reason": result.get("reason")}


async def memory_retriever_agent(
    task: str,
    context: dict,
    expected_response: str
) -> dict:
    sub_agent_builder = StateGraph(MemoryRetrieverState)
        
    # Add all modes
    sub_agent_builder.add_node("memory_search_tool_call_executor", memory_search_tool_call_executor)
    sub_agent_builder.add_node("memory_search_tool_call_generator", memory_search_tool_call_generator)
    sub_agent_builder.add_node("memory_search_tool_call_evaluator", memory_search_tool_call_evaluator)
    sub_agent_builder.add_node("memory_search_tool_call_finalizer", memory_search_tool_call_finalizer)

    # connect node
    sub_agent_builder.add_edge(START, "memory_search_tool_call_generator")

    # Linear flow: Generator -> Executor -> Evaluator
    sub_agent_builder.add_edge("memory_search_tool_call_generator", "memory_search_tool_call_executor")
    sub_agent_builder.add_edge("memory_search_tool_call_executor", "memory_search_tool_call_evaluator")

    # Evaluate
    sub_agent_builder.add_conditional_edges(
        "memory_search_tool_call_evaluator",
        lambda state: "loop" if not state["should_stop"] else "finalize",
        {
            "loop": "memory_search_tool_call_generator",
            "finalize": "memory_search_tool_call_finalizer",
        },
    )
    sub_agent_builder.add_edge("memory_search_tool_call_finalizer", END)

    sub_graph = sub_agent_builder.compile()

    # Build initial state
    subState: MemoryRetrieverState = {
        "task": task,
        "context": context,
        "next_tool": "",
        "expected_response": expected_response,
        "messages": [],
        "should_stop": False,
        "final_status": "error",
        "final_reason": "",
    }

    result = await sub_graph.ainvoke(subState)

    return {
        "status": result.get("final_status"),
        "reason": result.get("final_reason"),
        "response": result.get("final_response"),
        "knowledge_metadata": result.get("knowledge_metadata", {}),
    }


async def test_main():
    result = await memory_retriever_agent(
        task="千纱的角色设定",
        context={},
        expected_response="小于20字的答案"
    )

    print(result["status"])
    print(result["reason"])
    print(result["response"])
    print(result["knowledge_metadata"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_main())