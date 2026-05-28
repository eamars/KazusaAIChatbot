"""Memory evidence worker selection."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.evidence_coverage import has_explicit_multi_target_request
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

_KNOWN_WORKERS = {
    "persistent_memory_search_agent",
    "user_memory_evidence_agent",
    "incompatible",
}

def _strip_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value

def _memory_coverage_task(task: str) -> str:
    """Return the task text only when memory evidence needs strict coverage."""

    if has_explicit_multi_target_request(task):
        return_value = task
        return return_value
    return_value = ""
    return return_value

def _deterministic_plan(
    task: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Parse structured memory-evidence slots without selector LLM."""
    task_body = _strip_prefix(task)
    normalized = task_body.lower()

    if "active agreement" in normalized or "active promise" in normalized:
        plan = {
            "worker": "incompatible",
            "reason": "Recall",
        }
        return plan

    live_markers = (
        "current weather",
        "current temperature",
        "opening status",
        "current opening",
        "exchange rate",
        "current price",
    )
    if any(marker in normalized for marker in live_markers):
        plan = {
            "worker": "incompatible",
            "reason": "Live-context",
        }
        return plan

    person_markers = (
        "profile",
        "impression",
        "relationship",
        "compatibility",
    )
    if any(marker in normalized for marker in person_markers):
        plan = {
            "worker": "incompatible",
            "reason": "Person-context",
        }
        return plan

    slot_parts = [task_body]
    if isinstance(context, dict):
        current_slot = text_or_empty(context.get("current_slot"))
        if current_slot:
            slot_parts.append(current_slot)
    slot_text = "\n".join(slot_parts).lower()

    scoped_user_scope_markers = (
        "current user's",
        "current_user's",
        "current_user",
        "current user",
        "with the current user",
        "remember me",
        "recognize me",
    )
    ambiguous_user_scope_markers = (
        "user's",
        "users'",
        "user preferences",
        "user preference",
        "user decisions",
        "user decision",
    )
    scoped_user_topic_markers = (
        "continuity",
        "accepted preference",
        "accepted preferences",
        "preference",
        "preferences",
        "technical preference",
        "technical preferences",
        "shared experience",
        "past interaction",
        "past interactions",
        "prior interaction",
        "prior interactions",
        "interaction history",
        "prior shared interaction",
        "prior shared interactions",
        "shared history",
        "shared by current user",
        "shared by current_user",
        "current user's shared",
        "current-user's shared",
        "created by current user",
        "created by current_user",
        "current user's created",
        "current-user's created",
        "promise",
        "promises",
        "commitment",
        "commitments",
        "consideration",
        "considerations",
        "remember the current user",
        "recognize the current user",
        "remember me",
        "decision",
        "decisions",
        "choice",
        "choices",
        "care about",
        "cared about",
        "user memory evidence",
        "story lore",
        "story continuity",
        "private lore",
        "private continuity",
        "setting",
    )
    # Query-level context can confirm private scope, but each slot must carry
    # its own scoped-user topic so mixed queries keep independent memory paths.
    has_slot_scoped_user_scope = any(
        marker in slot_text for marker in scoped_user_scope_markers
    )
    has_scoped_user_topic = any(
        marker in slot_text for marker in scoped_user_topic_markers
    )
    if has_slot_scoped_user_scope and has_scoped_user_topic:
        plan = {
            "worker": "user_memory_evidence_agent",
            "reason": "scoped current-user continuity evidence",
        }
        return plan

    lifecycle_status_markers = (
        "completed",
        "outstanding",
        "fulfilled",
        "unfulfilled",
        "finished",
        "unfinished",
        "status",
    )
    has_runtime_user_scope = bool(
        isinstance(context, dict)
        and text_or_empty(context.get("global_user_id"))
    )
    has_lifecycle_status_topic = any(
        marker in slot_text
        for marker in lifecycle_status_markers
    )
    if has_runtime_user_scope and has_scoped_user_topic and has_lifecycle_status_topic:
        plan = {
            "worker": "user_memory_evidence_agent",
            "reason": "scoped current-user memory lifecycle evidence",
        }
        return plan

    has_ambiguous_user_scope = any(
        marker in slot_text for marker in ambiguous_user_scope_markers
    )
    if has_ambiguous_user_scope and has_scoped_user_topic:
        return None

    shared_memory_markers = (
        "official",
        "common sense",
        "world knowledge",
        "character-world",
        "character world",
        "character design",
        "home",
        "address",
        "location",
    )
    if any(marker in normalized for marker in shared_memory_markers):
        plan = {
            "worker": "persistent_memory_search_agent",
            "reason": "semantic durable memory evidence",
        }
        return plan

    exact_markers = (
        "named fact",
        "proper noun",
        "memory_name",
        "dedup_key",
        "tag",
        "exact",
        '"',
    )
    if any(marker in normalized for marker in exact_markers):
        plan = {
            "worker": "persistent_memory_search_agent",
            "reason": "hybrid durable named fact or exact memory evidence",
        }
        return plan

    if has_scoped_user_topic:
        return None

    if isinstance(context, dict) and text_or_empty(context.get("original_query")):
        return None

    plan = {
        "worker": "persistent_memory_search_agent",
        "reason": "semantic durable memory evidence",
    }
    return plan

def _normalize_selector_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM selector payload to approved fields."""
    worker = text_or_empty(raw_plan.get("worker"))
    if worker == "persistent_memory_keyword_agent":
        worker = "persistent_memory_search_agent"
    if worker not in _KNOWN_WORKERS:
        worker = "persistent_memory_search_agent"
    reason = text_or_empty(raw_plan.get("reason"))
    plan = {
        "worker": worker,
        "reason": reason,
    }
    return plan

_SELECTOR_PROMPT = '''\
你要为一个 durable evidence 槽位选择一个有边界的 persistent-memory worker。
不要用本路径回答活跃约定、人物资料、关系判断或实时外部事实。

# 生成步骤
1. 如果任务询问实时活跃约定、活跃承诺或当前 episode 状态，输出
   worker="incompatible"，reason="Recall"。历史已完成/未完成状态证据不适用此条。
2. 如果任务询问人物资料、印象、相性或关系上下文，输出
   worker="incompatible"，reason="Person-context"。
3. 如果任务询问当前天气、温度、营业状态、价格、汇率或任何变化中的实时值，
   输出 worker="incompatible"，reason="Live-context"。
4. 当前用户 durable memory、私有连续性、已接受偏好、用户专属设定、
   当前用户识别、过往互动历史、与当前用户的共同经历，以及过往用户专属
   promise/commitment 的已完成或未完成生命周期状态，使用
   user_memory_evidence_agent。
5. 自然语言 durable facts、精确命名事实、tags、memory_name/dedup_key 查询、
   proper nouns、quoted terms、home/address/location 问题、模糊概念、common sense、
   world knowledge 和 character-world facts，使用 persistent_memory_search_agent。
   该 worker 会执行语义加字面锚点的混合检索。

# 输入格式
{
  "task": "Memory-evidence 槽位文本",
  "original_query": "可用时的去上下文化用户问题",
  "current_slot": "槽位标签",
  "known_facts": "之前 RAG2 槽位得到的有序事实"
}

# 输出格式
只返回有效 JSON：
{
  "worker": "user_memory_evidence_agent | persistent_memory_search_agent | incompatible",
  "reason": "简短来源选择说明"
}
'''

_selector_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

async def _select_plan(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Select the bounded durable-memory worker path for one slot."""
    deterministic_plan = _deterministic_plan(task, context)
    if deterministic_plan is not None:
        return deterministic_plan

    llm_input = project_selector_input_for_llm(task, context)
    system_prompt = SystemMessage(content=_SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(llm_input, ensure_ascii=False, default=str)
    )
    response = await _selector_llm.ainvoke([system_prompt, human_message])
    raw_plan = parse_llm_json_output(response.content)
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    plan = _normalize_selector_plan(raw_plan)
    return plan
