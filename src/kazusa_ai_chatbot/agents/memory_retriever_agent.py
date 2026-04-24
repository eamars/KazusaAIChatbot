from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages # The magic ingredient

import json
import logging

from kazusa_ai_chatbot.config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, MAX_MEMORY_RETRIEVER_AGENT_RETRY
from kazusa_ai_chatbot.db import get_character_diary, get_objective_facts, search_conversation_history, get_conversation_history
from kazusa_ai_chatbot.db import search_memory as search_memory_db

from kazusa_ai_chatbot.utils import parse_llm_json_output, get_llm, sanitize_llm_text

from typing import Annotated, TypedDict

logger = logging.getLogger(__name__)


@tool
async def search_user_facts(global_user_id: str) -> list[str]:
    """Read user facts for a specific user.

    Mandatory argument rules:
    - global_user_id must be provided.
    - global_user_id must be the user's stable UUID string.
    
    Args:
        global_user_id (Mandatory): The global user ID (UUID) of the user.
        
    Returns:
        A list of user facts.
    """
    diary = await get_character_diary(global_user_id)
    facts = await get_objective_facts(global_user_id)
    return [e.get("entry", "") for e in diary if e.get("entry")] + \
           [f.get("fact", "") for f in facts if f.get("fact")]

@tool
async def search_conversation(search_query: str = "", 
                  global_user_id: str | None = None,
                  top_k: int = 5,
                  platform: str | None = None,
                  platform_channel_id: str | None = None,
                  from_timestamp: str | None = None,
                  to_timestamp: str | None = None,
    ) -> list[tuple[float, dict]]:
    """Search conversation history by semantic similarity.

    Mandatory argument rules:
    - search_query must be provided.
    - search_query must be a natural-language semantic query (not a keyword list).
    - Do not pass an empty string.
    
    Args:
        search_query (Mandatory): Semantic query sentence used for vector retrieval.
        global_user_id (Optional): Filter results to one user UUID.
        top_k (Optional): Maximum number of results to return. Default is 5.
        platform (Optional): Platform filter, e.g. "discord", "qq".
        platform_channel_id (Optional): Channel ID filter; if omitted, search all channels.
        from_timestamp (Optional): Start timestamp (ISO 8601).
        to_timestamp (Optional): End timestamp (ISO 8601).
        
    Returns:
        Top-K conversations close to the query, each as (similarity_score, message_with_metadata).
    """
    if not search_query or not search_query.strip():
        return [(-1.0, {"error": "search_query is mandatory and must not be empty. Please provide a natural-language semantic query."})]

    results = await search_conversation_history(
        query=search_query,
        platform=platform,
        platform_channel_id=platform_channel_id,
        global_user_id=global_user_id,
        limit=top_k,
        method="vector",
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )

    # Rebuild return format to remove unwanted columns
    return_list = []
    for (score, message) in results:
        return_list.append((score, {
            "content": message.get("content", ""),
            "timestamp": message.get("timestamp", ""),
            "display_name": message.get("display_name", ""),
            "role": message.get("role", ""),
            "platform": message.get("platform", ""),
            "platform_channel_id": message.get("platform_channel_id", ""),
            "platform_message_id": message.get("platform_message_id", ""),
            "platform_user_id": message.get("platform_user_id", ""),
            "global_user_id": message.get("global_user_id", ""),
            "reply_context": message.get("reply_context", {}),
        }))

    return return_list

@tool
async def search_conversation_keyword(
    keyword: str,
    global_user_id: str | None = None,
    top_k: int = 5,
    platform: str | None = None,
    platform_channel_id: str | None = None,
    from_timestamp: str | None = None,
    to_timestamp: str | None = None,
) -> list[dict]:
    """Search conversation history by exact keyword/phrase match (regex, case-insensitive).

    Use this tool when the search target is a specific term, technical phrase, or
    proper noun that must appear literally in the text (e.g. "HTTP", "DDR5").
    Prefer this over search_conversation when you know the exact wording.

    Args:
        keyword (Mandatory): Exact term or short phrase to match (regex, case-insensitive). Do not pass a full sentence — use the core noun/phrase only.
        global_user_id (Optional): Filter results to one user UUID.
        top_k (Optional): Maximum number of results. Default is 5.
        platform (Optional): Platform filter, e.g. "discord", "qq".
        platform_channel_id (Optional): Channel ID filter.
        from_timestamp (Optional): Start timestamp (ISO 8601).
        to_timestamp (Optional): End timestamp (ISO 8601).

    Returns:
        Matching conversations ordered by recency, each as a message dict.
    """
    if not keyword or not keyword.strip():
        return [{"error": "keyword is mandatory and must not be empty."}]

    results = await search_conversation_history(
        query=keyword,
        platform=platform,
        platform_channel_id=platform_channel_id,
        global_user_id=global_user_id,
        limit=top_k,
        method="keyword",
        from_timestamp=from_timestamp,
        to_timestamp=to_timestamp,
    )

    return [
        {
            "content": msg.get("content", ""),
            "timestamp": msg.get("timestamp", ""),
            "display_name": msg.get("display_name", ""),
            "role": msg.get("role", ""),
            "platform": msg.get("platform", ""),
            "platform_channel_id": msg.get("platform_channel_id", ""),
            "platform_message_id": msg.get("platform_message_id", ""),
            "platform_user_id": msg.get("platform_user_id", ""),
            "global_user_id": msg.get("global_user_id", ""),
            "reply_context": msg.get("reply_context", {}),
        }
        for _, msg in results
    ]


@tool
async def search_persistent_memory_keyword(
    keyword: str,
    top_k: int = 5,
    source_global_user_id: str | None = None,
    memory_type: str | None = None,
) -> list[dict]:
    """Search persistent memory by exact keyword/phrase match (regex, case-insensitive).

    Use this tool when the search target is a specific term, technical phrase, or
    proper noun that must appear literally in the stored memory content or name
    (e.g. "DDR5", "指令跟随逻辑"). Prefer this over search_persistent_memory when
    you know the exact wording.

    Mandatory argument rules:
    - keyword must be provided and should be the shortest unambiguous term or phrase.
    - Do not pass a full sentence — use the core noun/phrase only.

    Args:
        keyword (Mandatory): Exact term or short phrase to match (regex, case-insensitive).
        top_k (Optional): Maximum number of results. Default is 5.
        source_global_user_id (Optional): Filter by source user UUID.
        memory_type (Optional): Filter by type, e.g. "fact", "promise".

    Returns:
        Matching memory entries with metadata.
    """
    if not keyword or not keyword.strip():
        return [{"error": "keyword is mandatory and must not be empty."}]

    results = await search_memory_db(
        query=keyword,
        limit=top_k,
        method="keyword",
        source_global_user_id=source_global_user_id,
        memory_type=memory_type,
    )
    return [
        {
            "memory_name": mem.get("memory_name", ""),
            "content": mem.get("content", ""),
            "timestamp": mem.get("timestamp", ""),
            "source_global_user_id": mem.get("source_global_user_id", ""),
            "memory_type": mem.get("memory_type", ""),
            "status": mem.get("status", ""),
        }
        for _, mem in results
    ]


@tool
async def get_conversation(platform: str | None = None,
                           platform_channel_id: str | None = None,
                           limit: int = 5,
                           global_user_id: str | None = None,
                           display_name: str | None = None,
                           from_timestamp: str | None = None,
                           to_timestamp: str | None = None,
    ) -> list[dict]:
    """Get conversation history using structured filters.

    Usage rules:
    - At least one filter should be provided (for example platform_channel_id, global_user_id, or time range).
    - If both global_user_id and display_name are provided, global_user_id takes priority.
    - from_timestamp / to_timestamp should be ISO 8601 strings.
    
    Args:
        platform (Optional): Platform filter, e.g. "discord", "qq".
        platform_channel_id (Optional): Channel ID filter.
        limit (Optional): Maximum number of rows to return. Default is 5.
        global_user_id (Optional): User UUID filter.
        display_name (Optional): User display name filter (fallback if global_user_id is absent).
        from_timestamp (Optional): Start timestamp (ISO 8601), e.g. 2026-04-07T11:03:53.197223+00:00.
        to_timestamp (Optional): End timestamp (ISO 8601).
        
    Returns:
        A list of conversation messages.
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
            "display_name": message.get("display_name", ""),
            "role": message.get("role", ""),
            "platform": message.get("platform", ""),
            "platform_channel_id": message.get("platform_channel_id", ""),
            "platform_message_id": message.get("platform_message_id", ""),
            "platform_user_id": message.get("platform_user_id", ""),
            "global_user_id": message.get("global_user_id", ""),
            "reply_context": message.get("reply_context", {}),
        })

    return return_list


@tool
async def search_persistent_memory(
    search_query: str,
    top_k: int = 5,
    source_global_user_id: str | None = None,
    memory_type: str | None = None,
    source_kind: str | None = None,
    status: str | None = None,
    expiry_before: str | None = None,
    expiry_after: str | None = None,
) -> list[dict]:
    """Search persistent memory by semantic similarity and optional metadata filters.

    Mandatory argument rules:
    - search_query must be provided.
    - search_query must be a natural-language semantic query (not a keyword list).
    - Do not call this tool with only filters and no search_query.
    
    Args:
        search_query (Mandatory): Semantic query sentence for vector retrieval.
        top_k (Optional): Maximum number of results to return. Default is 5.
        source_global_user_id (Optional): Filter by source user UUID.
        memory_type (Optional): Filter by type, e.g. "fact", "promise", "impression", "narrative", "defense_rule".
        source_kind (Optional): Filter by source kind, e.g. "conversation_extracted", "seeded_manual".
        status (Optional): Filter by status, e.g. "active", "fulfilled", "expired", "superseded".
        expiry_before (Optional): ISO-8601 upper bound for expiry_timestamp (exclusive <).
        expiry_after (Optional): ISO-8601 lower bound for expiry_timestamp (exclusive >).

    Returns:
        Top-K memories close to the query, each with metadata and cosine similarity.
    """
    results = await search_memory_db(
        query=search_query,
        limit=top_k,
        method="vector",
        source_global_user_id=source_global_user_id,
        memory_type=memory_type,
        source_kind=source_kind,
        status=status,
        expiry_before=expiry_before,
        expiry_after=expiry_after,
    )

    # Rebuild return format to remove unwanted columns
    return_list = []
    for (score, memory) in results:
        return_list.append({
            "memory_name": memory.get("memory_name", ""),
            "content": memory["content"],
            "timestamp": memory["timestamp"],
            "source_global_user_id": memory.get("source_global_user_id", ""),
            "memory_type": memory.get("memory_type", ""),
            "source_kind": memory.get("source_kind", ""),
            "status": memory.get("status", ""),
            "confidence_note": memory.get("confidence_note", ""),
            "expiry_timestamp": memory.get("expiry_timestamp"),
            "cosine_similarity": score,
        })

    return return_list


_ALL_TOOLS = [
    search_user_facts,
    search_conversation,
    search_conversation_keyword,
    search_persistent_memory,
    search_persistent_memory_keyword,
    get_conversation,
]
_TOOLS_BY_NAME = {tool.name: tool for tool in _ALL_TOOLS}


def _inject_context_filters(tool_name: str, tool_args: dict, context: dict) -> dict:
    """Apply deterministic retrieval-scope filters from agent context to tool args.

    Args:
        tool_name: Name of the tool the LLM selected.
        tool_args: Raw tool arguments proposed by the LLM.
        context: Memory-retriever context dict supplied by the caller.

    Returns:
        A copy of ``tool_args`` with platform/channel/time-bound filters injected
        for conversation-history tools when the caller provided them.
    """
    args = dict(tool_args)
    if tool_name in {"search_conversation", "search_conversation_keyword", "get_conversation"}:
        target_platform = context.get("target_platform")
        if target_platform and "platform" not in args:
            args["platform"] = target_platform

        target_channel_id = context.get("target_platform_channel_id")
        if target_channel_id and "platform_channel_id" not in args:
            args["platform_channel_id"] = target_channel_id

        target_to_timestamp = context.get("target_to_timestamp")
        if target_to_timestamp and "to_timestamp" not in args:
            args["to_timestamp"] = target_to_timestamp

    return args


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
    final_is_empty_result: bool


async def memory_search_tool_call_executor(state: MemoryRetrieverState) -> dict:
    """Execute the tool calls generated by the LLM"""
    results = []
    last_message = state["messages"][-1]
    context = state.get("context", {}) or {}

    # Safety: Check if the LLM actually requested tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        for tool_call in last_message.tool_calls:
            try:
                tool = _TOOLS_BY_NAME[tool_call["name"]]
                tool_args = _inject_context_filters(tool_call["name"], tool_call["args"], context)
                observation = await tool.ainvoke(tool_args)
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

# 上下文过滤规则 (Context Filter Rules)
在发起任何工具调用前，必须先检查 `context` 中是否包含以下字段，并将其映射到对应工具参数：
- `target_platform` → 传入 `platform` 参数（如存在）。
- `target_platform_channel_id` → 传入 `platform_channel_id` 参数（如存在）。

**`target_global_user_id` 过滤规则（按查询类型区分）：**
- **类型 A（目标用户自身查询）**：将 `target_global_user_id` 传入所有支持该参数的工具（`search_conversation` 的 `global_user_id`、`search_persistent_memory` 的 `source_global_user_id`、`get_conversation` 的 `global_user_id`、`search_user_facts` 的 `global_user_id`），防止跨用户数据污染。
- **类型 B（第三方实体查询）**：`search_persistent_memory` 和 `search_conversation` **不传入** `source_global_user_id` / `global_user_id` 过滤器，因为第三方实体的记忆可能来源于不同用户，强制过滤会导致关键数据丢失。

**严禁**在类型 A 查询中未传入 `global_user_id` 过滤器的情况下调用 `search_conversation` 或 `search_persistent_memory`。

# 搜索方式选择规则 (Search Mode Rules)
每次搜索前，先判断使用**关键词搜索**还是**向量语义搜索**：

**关键词搜索** (`search_conversation_keyword` / `search_persistent_memory_keyword`)：
- 适用场景：目标是**确定存在于文本中的具体词语**，例如技术术语、专有名词、特定短语（”指令跟随逻辑”、”DDR5”、”学长”）。
- `keyword` 参数：尽量短且精确，只取核心名词/短语，**不得传入完整句子**。
- 正确示例：`keyword=”指令跟随逻辑”` / `keyword=”DDR5”`
- 错误示例：`keyword=”用户关于指令跟随逻辑的言论与观点”` （太长，应用向量搜索）

**向量语义搜索** (`search_conversation` / `search_persistent_memory`)：
- 适用场景：目标是**语义意图或印象**，例如用户的情感、态度、对某事物的看法，或无法确定用户是否用了准确词汇时。
- `search_query` 参数：用自然语言短语描述语义意图，嵌入”关系视角”或”话题背景”。
- 正确示例：`”用户对指令跟随逻辑的看法”` / `”Glitch 与杏山千纱的互动印象”`
- 错误示例：`”指令跟随逻辑”` （纯实体名词，应用关键词搜索）

**推荐策略**：
- 当 `task` 中出现明确的技术术语、专有名词或特定主题词时，**优先关键词搜索**；若关键词搜索无结果，再降级到向量搜索。
- 当 `task` 描述的是”印象”、”看法”、”评价”等语义内容时，直接使用向量搜索。
- 对于第三方实体（非 `target_global_user_id` 本人），关键词搜索时直接传入实体名称；向量搜索时必须将关系视角嵌入查询词（例如：`”杏山千纱对 X 的看法”` 而非仅 `”X”`）。

# 工具选择规则 (Tool Selection Rules)
在调用任何工具前，先判断查询类型：

**类型 A — 目标用户自身查询**：任务是检索 `target_global_user_id` 本人的信息（例如：用户自己的背景、偏好、承诺）。
- 工具优先级：`search_user_facts` → `search_conversation_keyword` / `search_persistent_memory_keyword` → `search_conversation` / `search_persistent_memory`
- `search_user_facts` 必须且只能在 `context` 中存在 `target_global_user_id` 时调用，传入该 UUID。

**类型 B — 第三方实体查询**：任务涉及在对话/记忆中提及的**人名或实体**（例如：”啾啾”、”Glitch”、某个朋友的名字）。
- **严禁**调用 `search_user_facts`（该工具不接受姓名，只接受用户 UUID）。
- 工具优先级：`search_conversation_keyword` / `search_persistent_memory_keyword` → `search_conversation` / `search_persistent_memory`

判断依据：若 `context.entities` 中的名称不等于 `target_global_user_id` 对应的用户名，即为类型 B。

# 任务流程
1. **分析历史**：审查 `messages`，确定已经执行过哪些查询。
2. **识别缺口**：对比 `task`，找出目前还缺失哪些关键信息。
3. **判断查询类型**：根据上方工具选择规则判断是类型 A 还是类型 B，选择对应工具集合。
4. **精准检索**：按所选类型的工具优先级顺序调用。若请求特定的聊天记录，则使用 `get_conversation`。
5. **调整策略**：如果之前的搜索返回空结果，必须更换关键词（例如：将”猫”改为”宠物”）或更换工具，禁止重复失败的操作。

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
请务必返回合法的 JSON 字符串，仅包含以下字段：
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
        content=json.dumps(
            {
                "feedback": feedback,
                "source": "evaluator",
            },
            ensure_ascii=False,
        ),
        name="evaluator"
    )

    return {
        "messages": [feedback_message],
        "should_stop": should_stop,
        "retry": retry,
        "knowledge_metadata": knowledge_metadata,
    }



_MEMORY_RETRIEVER_FINALIZER_PROMPT = """\
你是一个上下文整合专家。你的输出将直接作为下游 LLM 代理的输入上下文，因此信息完整性优先于简洁性。

# 核心任务
1. **完整保留**：将所有与任务相关的检索内容完整写入 `response`。严禁以"简洁"为由丢弃有效信息——下游代理需要原始事实，而不是你的摘要。
2. **清理噪音**：过滤掉明显与任务无关的内容（例如：与 Kazusa 完全无关的群聊闲聊），但保留所有与目标用户和 Kazusa 互动相关的记录。
3. **保留说话者元数据**：如果检索结果中提供了 `display_name` / `role` / `platform_user_id`，渲染对话记录时必须优先使用这些显式字段标注说话者；**禁止**根据 `content` 中出现的 `<@...>` 提及、reply 目标或其它文本线索去猜测说话者身份。
4. **解析标识符**：你只能替换 `content` 正文中出现的平台 ID（如 `<@3768713357>`、UUID 格式的用户 ID）为可读名称。规则：若 UUID 与 `context.target_global_user_id` 匹配，替换为 `context.target_user_name`；若 ID 与平台 bot ID 匹配，替换为 "Kazusa"。这条规则只作用于消息正文，不作用于说话者标签。
5. **评估完整度**：根据评估者最终反馈与任务要求，客观评分。

# 输出说明
- response: 整合后的完整上下文字符串，供下游 LLM 代理直接使用。包含所有相关事实、对话记录和记忆条目，并附带时间戳和来源说明。
- score: 0-100，表示检索内容满足任务需求的程度
- reason: 一句话说明评分依据
- is_empty_result: 布尔值。仅当最终确认没有任何任务相关检索内容可供下游使用时为 true；只要存在任何任务相关事实，即使数量很少，也必须为 false。
# 输入格式
{
    "task": "任务描述",
    "content": "所有工具返回的原始数据",
    "evaluator_feedback": "评估者最终反馈",
    "expected_response": "下游代理期望的输出内容和格式",
    "context": { "target_user_name": "目标用户可读名称", "target_global_user_id": "目标用户 UUID", ... }
}

# 输出格式
请务必返回合法的 JSON 字符串，仅包含以下字段：
{
    "response": "string",
    "score": <int: 0-100>,
    "reason": "string",
    "is_empty_result": true or false
}
"""
_memory_search_tool_call_finalizer_llm = get_llm(temperature=0.0, top_p=1.0)
_MAX_TOOL_RESULTS_CHARS = 12_000

async def memory_search_tool_call_finalizer(state: MemoryRetrieverState) -> dict:
    """Finalize the retrieved info into the expected format"""
    # Collect tool results — sanitize control chars that cause API 400 rejections
    tool_messages = [sanitize_llm_text(m.content) for m in state["messages"] if isinstance(m, ToolMessage)]
    tool_results = "\n".join(tool_messages) if tool_messages else "No information retrieved."

    # Cap size to avoid exceeding API limits
    if len(tool_results) > _MAX_TOOL_RESULTS_CHARS:
        tool_results = tool_results[:_MAX_TOOL_RESULTS_CHARS] + "\n...[truncated]"

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
        "context": state.get("context", {}),
    }
    human_message = HumanMessage(content=json.dumps(finalizer_input, ensure_ascii=False))

    try:
        response = await _memory_search_tool_call_finalizer_llm.ainvoke([system_prompt, human_message])
        result = parse_llm_json_output(response.content)
    except Exception as e:
        logger.error(f"Finalizer LLM call failed: {e}")
        return {
            "final_response": "No information retrieved.",
            "final_status": "error",
            "final_reason": f"Finalizer failed: {type(e).__name__}",
            "final_is_empty_result": True,
        }

    # Status generation
    status = ""
    if "score" in result:
        if result["score"] > 80:
            status = "complete"
        elif result["score"] > 50:
            status = "partial"
        elif result["score"] > 0:
            status = "incomplete"
    else:
        # Handle the case where status is not returned
        status = "error"

    # Do some sanity check
    if "response" not in result:
        result["response"] = "No information retrieved."
        result["score"] = 0
        status = "error"
        logger.error(f"No response provided by finalizer, raw result: \n{result}")

    if "reason" not in result:
        result["reason"] = "No reason provided."
    

    # final_message = AIMessage(content=result.get("response", ""))
    is_empty_result = result.get("is_empty_result")
    if not isinstance(is_empty_result, bool):
        logger.error(
            "Memory retriever finalizer omitted is_empty_result; raw result=%s",
            result,
        )
        is_empty_result = False
     
 
    return {"final_response": result.get("response"), 
            "final_status": status, 
            "final_reason": result.get("reason"),
            "final_is_empty_result": is_empty_result}


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
        "final_is_empty_result": False,
    }

    result = await sub_graph.ainvoke(subState)

    return {
        "status": result.get("final_status"),
        "reason": result.get("final_reason"),
        "response": result.get("final_response"),
        "is_empty_result": result.get("final_is_empty_result", False),
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
