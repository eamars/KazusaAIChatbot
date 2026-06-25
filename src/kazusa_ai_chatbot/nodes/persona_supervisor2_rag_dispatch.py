"""Dispatcher, agent registry, and executor for the RAG supervisor."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from kazusa_ai_chatbot import event_logging, llm_tracing
from kazusa_ai_chatbot.config import (

    RAG_PLANNER_LLM_API_KEY,
    RAG_PLANNER_LLM_BASE_URL,
    RAG_PLANNER_LLM_MODEL,
    RAG_PLANNER_LLM_MAX_COMPLETION_TOKENS,
    RAG_PLANNER_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.conversation_evidence.workers import (
    ConversationAggregateAgent,
    ConversationFilterAgent,
    ConversationKeywordAgent,
    ConversationSearchAgent,
)
from kazusa_ai_chatbot.rag.live_context import LiveContextAgent
from kazusa_ai_chatbot.rag.memory_evidence import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.memory_evidence.workers import (
    PersistentMemoryKeywordAgent,
    PersistentMemorySearchAgent,
)
from kazusa_ai_chatbot.rag.person_context import PersonContextAgent
from kazusa_ai_chatbot.rag.person_context.workers import (
    RelationshipAgent,
    UserListAgent,
    UserLookupAgent,
    UserProfileAgent,
)
from kazusa_ai_chatbot.rag.prompt_projection import project_runtime_context_for_llm
from kazusa_ai_chatbot.rag.recall import RecallAgent
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_prompt_views import (
    _known_facts_llm_view,
)
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_types import (
    ProgressiveRAGState,
    RAGAgentRegistryEntry,
)
from kazusa_ai_chatbot.utils import (
    log_preview,
    parse_llm_json_output,
)

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
RAG_DISPATCH_COMPONENT = "nodes.persona_supervisor2_rag_dispatch"


def _elapsed_ms(started_at: float) -> int:
    """Return elapsed monotonic milliseconds since a start marker."""

    elapsed = time.perf_counter() - started_at
    elapsed_ms = max(0, int(elapsed * MILLISECONDS_PER_SECOND))
    return elapsed_ms


def _state_correlation_id(state: ProgressiveRAGState) -> str:
    """Build a non-content correlation id for RAG dispatch work."""

    context = state.get("context", {})
    if isinstance(context, dict):
        platform = str(context.get("platform", ""))
        message_ref = str(context.get("platform_message_id", "") or "")
    else:
        platform = ""
        message_ref = ""
    correlation_id = f"rag:{platform}:{message_ref or 'no-message-id'}"
    return correlation_id


# ── Dispatcher ─────────────────────────────────────────────────────
_RAG_SUPERVISOR_AGENT_REGISTRY: dict[str, RAGAgentRegistryEntry] = {
    "live_context_agent": {
        "agent": LiveContextAgent().run,
        "fact_source": {
            "source_kind": "external",
            "source_system": "live_context",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_evidence_agent": {
        "agent": ConversationEvidenceAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "memory_evidence_agent": {
        "agent": MemoryEvidenceAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "person_context_agent": {
        "agent": PersonContextAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_lookup_agent": {
        "agent": UserLookupAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_list_agent": {
        "agent": UserListAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles_or_conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "user_profile_agent": {
        "agent": UserProfileAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "relationship_agent": {
        "agent": RelationshipAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "user_profiles",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "web_agent3": {
        "agent": WebAgent3().run,
        "fact_source": {
            "source_kind": "external",
            "source_system": "web",
            "consolidation_policy": "eligible_external_knowledge",
            "can_consolidate_as_new_knowledge": True,
        },
    },
    "conversation_aggregate_agent": {
        "agent": ConversationAggregateAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_filter_agent": {
        "agent": ConversationFilterAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_keyword_agent": {
        "agent": ConversationKeywordAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "conversation_search_agent": {
        "agent": ConversationSearchAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "conversation_history",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "persistent_memory_keyword_agent": {
        "agent": PersistentMemoryKeywordAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "persistent_memory_search_agent": {
        "agent": PersistentMemorySearchAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "memory",
            "consolidation_policy": "do_not_write_knowledge",
            "can_consolidate_as_new_knowledge": False,
        },
    },
    "recall_agent": {
        "agent": RecallAgent().run,
        "fact_source": {
            "source_kind": "internal",
            "source_system": "recall",
            "consolidation_policy": "operational_recall_evidence",
            "can_consolidate_as_new_knowledge": False,
        },
    },
}

_PREFIX_DISPATCH_TABLE: tuple[tuple[str, str, int], ...] = (
    ("Live-context:", "live_context_agent", 1),
    ("Conversation-evidence:", "conversation_evidence_agent", 1),
    ("Memory-evidence:", "memory_evidence_agent", 1),
    ("Person-context:", "person_context_agent", 1),
    ("Web-evidence:", "web_agent3", 3),
    ("Identity:", "user_lookup_agent", 3),
    ("User-list:", "user_list_agent", 3),
    ("Relationship:", "relationship_agent", 3),
    ("Profile:", "user_profile_agent", 3),
    ("Conversation-aggregate:", "conversation_evidence_agent", 1),
    ("Conversation-filter:", "conversation_evidence_agent", 1),
    ("Conversation-keyword:", "conversation_evidence_agent", 1),
    ("Conversation-semantic:", "conversation_evidence_agent", 1),
    ("Memory-keyword:", "memory_evidence_agent", 1),
    ("Memory-search:", "memory_evidence_agent", 1),
    ("Recall:", "recall_agent", 1),
    ("Web-search:", "web_agent3", 3),
)

_DISPATCH_AGENT_ALIASES = {
    "conversation_aggregate_agent": "conversation_evidence_agent",
    "conversation_filter_agent": "conversation_evidence_agent",
    "conversation_keyword_agent": "conversation_evidence_agent",
    "conversation_search_agent": "conversation_evidence_agent",
    "persistent_memory_keyword_agent": "memory_evidence_agent",
    "persistent_memory_search_agent": "memory_evidence_agent",
}

_DISPATCHER_PROMPT = '''\
你是 RAG Dispatcher。对每个槽位，选择且只选择一个 inner-loop retrieval agent，
并为该代理生成简短任务描述。

## Agent roster

- `live_context_agent`：顶层当前时态 live context 能力。直接回答 runtime 支持的
  当前本地时间/日期/星期；对天气、温度、营业状态、日程、价格、汇率或当前公开状态，
  先解析 target/scope，再委托给 web。用于 `Live-context:` 槽位。

- `conversation_evidence_agent`：顶层聊天历史证据能力。内部选择混合精确/模糊搜索、
  结构化过滤或聚合 worker。用于 `Conversation-evidence:` 以及 legacy
  `Conversation-keyword:`、`Conversation-semantic:`、`Conversation-filter:`、
  `Conversation-aggregate:` 槽位。

- `memory_evidence_agent`：顶层 durable memory evidence 能力。处理回答槽位所需的
  durable memory evidence，并内部选择混合精确/模糊 persistent-memory workers。
  用于 `Memory-evidence:`、legacy `Memory-search:`、`Memory-keyword:` 槽位。

- `person_context_agent`：顶层 person/profile/relationship 能力。内部选择 identity、
  profile、user-list 或 relationship worker。用于 `Person-context:` 槽位。

- `user_lookup_agent`：按 display name 直接查询 user_profiles，返回 global_user_id。
  仅当槽位本身是在解析命名人物是谁时使用。

- `user_list_agent`：按 display-name 谓词或 participant metadata 枚举用户。
  处理 display name 等于、包含、开头或结尾为某个字面值的用户列表问题。

- `user_profile_agent`：读取用户完整 profile。仅当 known_facts 已有 global_user_id
  时使用；不要用于未知身份。

- `relationship_agent`：按活跃角色的关系数据对已建档用户排序。
  用于 `Relationship:` 槽位；代理自行抽取排名参数。

- `recall_agent`：从 scoped progress、active commitments、pending scheduled events
  和 gated history proof 中协调活跃约定、持续承诺、当前计划、open loops 和当前 episode 状态。
  用于 `Recall:` 槽位。

- `web_agent3`：公开互联网搜索。仅当信息不可能存在于本地聊天历史或 persistent memory 时使用。

内部 worker 由顶层能力自行选择，不要把它们作为 `agent_name` 输出。

## 槽位前缀 → agent mapping（先检查；覆盖下面所有规则）

initializer 生成的槽位以可直接映射到 agent 的前缀开头。
按字面匹配前缀，并直接使用映射的 agent，不要额外推理：

| Slot prefix                  | Agent                            |
|------------------------------|----------------------------------|
| "Live-context: ..."          | `live_context_agent`             |
| "Conversation-evidence: ..." | `conversation_evidence_agent`    |
| "Memory-evidence: ..."       | `memory_evidence_agent`          |
| "Person-context: ..."        | `person_context_agent`           |
| "Web-evidence: ..."          | `web_agent3`                     |
| "Identity: ..."              | `user_lookup_agent`              |
| "User-list: ..."             | `user_list_agent`                |
| "Relationship: ..."          | `relationship_agent`             |
| "Profile: ..."               | `user_profile_agent`             |
| "Conversation-aggregate: ..."| `conversation_evidence_agent`    |
| "Conversation-filter: ..."   | `conversation_evidence_agent`    |
| "Conversation-keyword: ..."  | `conversation_evidence_agent`    |
| "Conversation-semantic: ..." | `conversation_evidence_agent`    |
| "Memory-keyword: ..."        | `memory_evidence_agent`          |
| "Memory-search: ..."         | `memory_evidence_agent`          |
| "Recall: ..."                | `recall_agent`                   |
| "Web-search: ..."            | `web_agent3`                     |

## Fallback decision sequence（仅当槽位没有可识别前缀时使用）

从上到下检查，选择第一个匹配项：

1. 槽位解析命名人物身份（display_name → global_user_id）？
   → `user_lookup_agent`.

2. 槽位按 display-name pattern 或 participant metadata 枚举用户？
   → `user_list_agent`.

3. 槽位需要用户完整 profile，且 known_facts 已有 global_user_id？
   → `user_profile_agent`.

4. 槽位目标是聊天历史，包括字面字符串、URLs、filenames、精确短语、模糊话题、
   结构化过滤、计数或排名？
   → `conversation_evidence_agent`.

5. 槽位目标是 durable persistent memory，包括精确关键词、tag、event name、
   proper noun、memory identifier 或语义记忆含义？
   → `memory_evidence_agent`.

6. 槽位询问约定、承诺、计划、未解决事项，或当前 episode 停在何处？
   → `recall_agent`.

7. 槽位需要公开互联网数据？
   → `web_agent3`.

## 输入格式

{{
    "current_slot": "本轮要解决的槽位",
    "known_facts": [{{"slot": "已完成槽位", "agent": "agent_name", "resolved": true, "summary": "简短事实摘要", "attempts": 1}}],
    "context": {{"platform": "qq", "channel": "频道标识", "timestamp": "当前时间"}}
}}

## 生成步骤
1. 读取 `current_slot`，先与 prefix-to-agent 表做字面匹配。
2. 如果没有可识别前缀，使用 fallback decision sequence。
3. 为选中的 inner-loop agent 生成简短任务，保留 "slot N" 这样的依赖引用。
4. 只传递该 agent 需要的相关 context。

## 输出格式

只返回有效 JSON：
{{
    "agent_name": "{agent_name_union}",
    "task": "给所选 agent 的任务描述",
    "context": {{}},
    "max_attempts": 3
}}
'''
_llm_interface = LLInterface()
_dispatcher_llm = LLInterface()
_dispatcher_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="RAG_PLANNER_LLM",
    base_url=RAG_PLANNER_LLM_BASE_URL,
    api_key=RAG_PLANNER_LLM_API_KEY,
    model=RAG_PLANNER_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=RAG_PLANNER_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=RAG_PLANNER_LLM_THINKING_ENABLED,
    ),
)


def build_rag_fact_source_map() -> dict[str, dict[str, Any]]:
    """Build the deterministic fact-source map for RAG known_facts.

    The consolidator can combine this map with each known fact's ``agent``
    field to decide whether a fact came from internal memory/history/profile
    stores or from an external source.

    Returns:
        Dict keyed by agent name with source and consolidation policy metadata.
    """
    return_value = {
        agent_name: dict(entry["fact_source"])
        for agent_name, entry in _RAG_SUPERVISOR_AGENT_REGISTRY.items()
    }
    return return_value


def _dispatch_from_known_prefix(current_slot: str) -> dict[str, Any] | None:
    """Build a dispatch directly from a recognized slot prefix.

    Args:
        current_slot: The ordered slot selected for this loop iteration.

    Returns:
        Normalized dispatch payload, or ``None`` when no prefix matches.
    """

    if not isinstance(current_slot, str):
        return None

    for prefix, agent_name, max_attempts in _PREFIX_DISPATCH_TABLE:
        if current_slot.startswith(prefix):
            dispatch = {
                "agent_name": agent_name,
                "task": current_slot.strip(),
                "context": {},
                "max_attempts": max_attempts,
                "route_source": "deterministic_prefix",
            }
            return dispatch

    return_value = None
    return return_value


async def rag_dispatcher(state: ProgressiveRAGState) -> dict:
    """Generate one plain JSON dispatch targeting the first unknown slot.

    Args:
        state: Current state with unknown_slots and known_facts.

    Returns:
        Partial state update with dispatcher JSON recorded in state,
        current_slot set, and loop_count incremented.
    """
    current_slot = state["unknown_slots"][0]
    prefix_dispatch = _dispatch_from_known_prefix(current_slot)
    if prefix_dispatch is not None:
        logger.info(
            f'RAG2 dispatch output: agent={prefix_dispatch["agent_name"]} '
            f"task={log_preview(prefix_dispatch['task'])} "
            f'route_source={prefix_dispatch["route_source"]}'
        )
        logger.debug(
            f'RAG2 dispatch metadata: loop={state.get("loop_count", 0) + 1} '
            f"slot={log_preview(current_slot)} "
            f'max_attempts={prefix_dispatch["max_attempts"]} '
            f"dispatch_context={log_preview(prefix_dispatch['context'])}"
        )
        return_value = {
            "messages": [
                AIMessage(
                    content=json.dumps(prefix_dispatch, ensure_ascii=False)
                )
            ],
            "current_slot": current_slot,
            "current_dispatch": prefix_dispatch,
            "loop_count": state.get("loop_count", 0) + 1,
        }
        return return_value

    system_prompt = SystemMessage(
        content=_DISPATCHER_PROMPT.format(
            agent_name_union=_build_agent_name_union(),
        )
    )
    llm_context = project_runtime_context_for_llm(
        state.get("context", {}),
        character_name=state.get("character_name", ""),
    )
    user_input = {
        "current_slot": current_slot,
        "known_facts": _known_facts_llm_view(state.get("known_facts", [])),
        "context": llm_context,
    }
    human_message = HumanMessage(content=json.dumps(user_input, ensure_ascii=False, default=str))

    # Only the last two messages — enough to preserve immediate loop history
    # without polluting the current slot with distant attempts.
    recent_messages = state["messages"][-2:] if len(state["messages"]) >= 2 else state["messages"]

    started_at = time.perf_counter()
    response = await _dispatcher_llm.ainvoke([system_prompt, human_message] + recent_messages, config=_dispatcher_llm_config)
    raw_dispatch = parse_llm_json_output(response.content)
    parse_status = "succeeded" if isinstance(raw_dispatch, dict) else "failed"
    if not isinstance(raw_dispatch, dict):
        dispatch = {}
    else:
        dispatch = raw_dispatch
    normalized_dispatch = _normalize_dispatch(dispatch, current_slot)
    if not normalized_dispatch["agent_name"]:
        parse_status = "invalid_contract"
    logger.info(
        f'RAG2 dispatch output: agent={normalized_dispatch["agent_name"] or "<invalid>"} '
        f"task={log_preview(normalized_dispatch['task'])} "
        f'route_source={normalized_dispatch["route_source"]}'
    )
    logger.debug(
        f'RAG2 dispatch metadata: loop={state.get("loop_count", 0) + 1} '
        f"slot={log_preview(current_slot)} "
        f'max_attempts={normalized_dispatch["max_attempts"]} '
        f'route_source={normalized_dispatch["route_source"]} '
        f"dispatch_context={log_preview(normalized_dispatch['context'])} "
        f"raw={log_preview(dispatch)}"
    )
    await event_logging.record_llm_stage_event(
        component=RAG_DISPATCH_COMPONENT,
        stage_name="rag_dispatcher",
        route_name=normalized_dispatch["route_source"],
        model_name=RAG_PLANNER_LLM_MODEL,
        status="succeeded" if normalized_dispatch["agent_name"] else "failed",
        prompt_chars=len(system_prompt.content) + len(human_message.content),
        output_chars=len(str(response.content)),
        parse_status=parse_status,
        retry_count=0,
        json_repair_used=False,
        correlation_id=_state_correlation_id(state),
        duration_ms=_elapsed_ms(started_at),
        severity="info" if normalized_dispatch["agent_name"] else "warning",
    )
    if not normalized_dispatch["agent_name"]:
        await event_logging.record_model_contract_event(
            component=RAG_DISPATCH_COMPONENT,
            stage_name="rag_dispatcher",
            violation_kind="invalid_dispatch",
            missing_fields=[],
            invalid_fields=["agent_name"],
            repair_used=False,
            status="failed",
            correlation_id=_state_correlation_id(state),
        )

    return_value = {
        "messages": [AIMessage(content=response.content)],
        "current_slot": current_slot,
        "current_dispatch": normalized_dispatch,
        "loop_count": state.get("loop_count", 0) + 1,
    }
    await llm_tracing.record_llm_trace_step(
        trace_id=str(state.get("llm_trace_id", "")),
        stage_name="rag_dispatcher",
        route_name=normalized_dispatch["route_source"],
        model_name=RAG_PLANNER_LLM_MODEL,
        messages=[system_prompt, human_message] + recent_messages,
        response_text=str(response.content),
        parsed_output=return_value,
        parse_status=parse_status,
        status="succeeded" if normalized_dispatch["agent_name"] else "failed",
        duration_ms=_elapsed_ms(started_at),
        output_state_fields=[
            "messages",
            "current_slot",
            "current_dispatch",
            "loop_count",
        ],
    )
    return return_value


# ── Executor ───────────────────────────────────────────────────────


def _build_delegate_context(state: ProgressiveRAGState, dispatch: dict) -> dict:
    """Merge runtime context, dispatcher hints, and known facts for one agent call."""
    delegate_context = dict(state.get("context", {}))
    raw_dispatch_context = dispatch.get("context", {})
    if isinstance(raw_dispatch_context, dict):
        delegate_context.update(raw_dispatch_context)
    delegate_context["known_facts"] = _known_facts_llm_view(
        state.get("known_facts", []),
    )
    delegate_context["original_query"] = state.get("original_query", "")
    delegate_context["current_slot"] = state.get("current_slot", "")
    return delegate_context



def _build_agent_name_union() -> str:
    """Render the dispatcher's allowed ``agent_name`` values as a union string."""
    agent_names = [
        agent_name
        for agent_name in _RAG_SUPERVISOR_AGENT_REGISTRY
        if agent_name not in _DISPATCH_AGENT_ALIASES
    ]
    return_value = " | ".join(agent_names)
    return return_value


def _normalize_dispatch(raw_dispatch: dict, current_slot: str) -> dict:
    """Normalize dispatcher JSON into a safe executable dispatch payload."""
    raw_agent_name = raw_dispatch.get("agent_name", "")
    agent_name = raw_agent_name.strip() if isinstance(raw_agent_name, str) else ""
    agent_name = _DISPATCH_AGENT_ALIASES.get(agent_name, agent_name)
    if agent_name not in _RAG_SUPERVISOR_AGENT_REGISTRY:
        agent_name = ""

    raw_task = raw_dispatch.get("task", "")
    fallback_task = current_slot if isinstance(current_slot, str) else ""
    task = raw_task.strip() if isinstance(raw_task, str) else ""
    task = task or fallback_task
    context = raw_dispatch.get("context", {})
    if not isinstance(context, dict):
        context = {}

    raw_max_attempts = raw_dispatch.get("max_attempts", 3)
    if isinstance(raw_max_attempts, int) and not isinstance(raw_max_attempts, bool) and raw_max_attempts > 0:
        max_attempts = raw_max_attempts
    else:
        max_attempts = 3

    raw_route_source = raw_dispatch.get("route_source", "dispatcher_llm")
    if isinstance(raw_route_source, str) and raw_route_source.strip():
        route_source = raw_route_source.strip()
    else:
        route_source = "dispatcher_llm"

    return_value = {
        "agent_name": agent_name,
        "task": task,
        "context": context,
        "max_attempts": max_attempts,
        "route_source": route_source,
    }
    return return_value


def _serialize_agent_result(result: dict) -> str:
    """Serialize one agent result for dispatcher history."""
    return_value = json.dumps(result, ensure_ascii=False, default=str)
    return return_value


def _agent_result_info_view(result: dict) -> dict:
    """Build an INFO-safe view of an agent result.

    Args:
        result: Raw helper-agent result from the executor.

    Returns:
        Compact operational fields suitable for INFO logging. Heavy payloads
        such as resolved refs, projection payloads, and worker payloads remain
        available in DEBUG logs and in graph state.
    """

    raw_result = result.get("result")
    if isinstance(raw_result, dict) and "capability" in raw_result:
        compact_result = {
            "capability": raw_result.get("capability", ""),
            "primary_worker": raw_result.get("primary_worker", ""),
            "supporting_workers": raw_result.get("supporting_workers", []),
            "missing_context": raw_result.get("missing_context", []),
            "selected_summary": raw_result.get("selected_summary", ""),
        }
    else:
        compact_result = raw_result

    raw_cache = result.get("cache", {})
    cache = raw_cache if isinstance(raw_cache, dict) else {}
    info_view = {
        "agent": result.get("agent", ""),
        "resolved": bool(result.get("resolved", False)),
        "attempts": result.get("attempts", 0),
        "cache": {
            "enabled": cache.get("enabled", False),
            "hit": cache.get("hit", False),
            "reason": cache.get("reason", ""),
        },
        "result": compact_result,
    }
    return info_view


async def rag_executor(state: ProgressiveRAGState) -> dict:
    """Execute the delegate-agent call produced by the dispatcher.

    Args:
        state: Current state whose dispatch payload names the agent to run.

    Returns:
        Partial state update with the normalized agent result.
    """
    dispatch = _normalize_dispatch(
        state.get("current_dispatch", {}),
        state.get("current_slot", ""),
    )
    agent_name = dispatch["agent_name"]

    if not agent_name:
        result = {
            "agent": "",
            "resolved": False,
            "result": "Dispatcher failed to choose a valid agent.",
            "attempts": 0,
        }
        logger.info(
            f"RAG2 agent output: result={log_preview(result)}"
        )
        logger.debug(
            f'RAG2 agent metadata: slot={log_preview(state.get("current_slot", ""))} '
            f"agent=<invalid> resolved=False attempts=0"
        )
        return_value = {
            "last_agent_result": result,
            "messages": [
                HumanMessage(
                    content=_serialize_agent_result(_agent_result_info_view(result)),
                    name="agent_result",
                )
            ],
        }
        return return_value

    agent = _RAG_SUPERVISOR_AGENT_REGISTRY[agent_name]["agent"]
    try:
        result = await agent(
            task=dispatch["task"],
            context=_build_delegate_context(state, dispatch),
            max_attempts=dispatch["max_attempts"],
        )
        if not isinstance(result, dict):
            result = {
                "resolved": False,
                "result": str(result),
                "attempts": dispatch["max_attempts"],
            }
        result.setdefault("agent", agent_name)
    except Exception as exc:
        logger.exception(f'Error executing agent {agent_name}: {exc}')
        result = {
            "agent": agent_name,
            "resolved": False,
            "result": f"{type(exc).__name__}: {exc}",
            "attempts": 0,
        }

    info_view = _agent_result_info_view(result)
    logger.info(f"RAG2 agent output: result={log_preview(info_view)}")
    logger.debug(
        f'RAG2 agent metadata: slot={log_preview(state.get("current_slot", ""))} '
        f'agent={agent_name} resolved={bool(result.get("resolved", False))} '
        f'attempts={result.get("attempts", 0)} raw={log_preview(result)}'
    )
    return_value = {
        "last_agent_result": result,
        "messages": [
            HumanMessage(
                content=_serialize_agent_result(info_view),
                name="agent_result",
            )
        ],
    }
    return return_value
