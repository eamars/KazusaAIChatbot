"""Live context plan selection."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (

    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.rag.live_context.target_resolution import (
    _clean_target,
    _target_after_marker,
)
from kazusa_ai_chatbot.rag.live_context.runtime_facts import _RUNTIME_FACT_TYPES
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
_KNOWN_FACT_TYPES = {
    "current_time",
    "current_date",
    "current_weekday",
    "weather",
    "temperature",
    "opening_status",
    "schedule",
    "price",
    "exchange_rate",
    "current_event_status",
    "other",
}

_KNOWN_TARGET_SOURCES = {
    "explicit",
    "active_character_default",
    "current_user_recent",
    "unknown",
}

def _extract_fact_type(task: str) -> str:
    """Classify the live fact type from structured slot text.

    Args:
        task: Initializer slot or dispatcher task text.

    Returns:
        One approved live fact type label.
    """

    normalized = task.lower()
    if "local weekday" in normalized:
        return "current_weekday"
    if "local date" in normalized:
        return "current_date"
    if "local time" in normalized:
        return "current_time"
    if "current local time" in normalized or "current time" in normalized:
        return "current_time"
    if "current local date" in normalized or "current date" in normalized:
        return "current_date"
    if "current local weekday" in normalized or "current weekday" in normalized:
        return "current_weekday"
    if "temperature" in normalized:
        return "temperature"
    if "weather" in normalized:
        return "weather"
    if "opening status" in normalized or "open status" in normalized:
        return "opening_status"
    if "open " in normalized or "is open" in normalized:
        return "opening_status"
    if "schedule" in normalized or "timetable" in normalized:
        return "schedule"
    if "price" in normalized or "cost" in normalized:
        return "price"
    if "exchange rate" in normalized:
        return "exchange_rate"
    if "current event" in normalized or "event status" in normalized:
        return "current_event_status"
    return "other"

def _strip_live_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value

def _deterministic_plan(task: str) -> dict[str, Any] | None:
    """Parse structured live-context slots without a selector LLM.

    Args:
        task: Slot text supplied by the RAG2 dispatcher.

    Returns:
        A selector plan when the slot contains approved structured markers,
        otherwise ``None`` so the selector LLM can classify the request.
    """

    task_body = _strip_live_prefix(task)
    normalized = task_body.lower()
    fact_type = _extract_fact_type(task_body)

    if fact_type in _RUNTIME_FACT_TYPES:
        target = _target_after_marker(task_body, "explicit location")
        if not target:
            target = _target_after_marker(task_body, "explicit target")
        if not target and " for " in normalized:
            _, _, extracted_target = task_body.rpartition(" for ")
            extracted_target = _clean_target(extracted_target)
            extracted_target_lower = extracted_target.lower()
            if extracted_target and extracted_target_lower not in {
                "unknown location",
                "unknown target",
            }:
                if "location" not in extracted_target_lower:
                    target = extracted_target

        if target:
            plan = {
                "source_class": "external_live_lookup",
                "runtime_scope": "",
                "fact_type": fact_type,
                "target_source": "explicit",
                "target": target,
                "missing_context": [],
            }
            return plan

        runtime_scope = "active_character"
        if "current user" in normalized:
            runtime_scope = "current_user"

        plan = {
            "source_class": "runtime_snapshot",
            "runtime_scope": runtime_scope,
            "fact_type": fact_type,
            "target_source": "unknown",
            "target": "",
            "missing_context": [],
        }
        return plan

    if "current user's location" in normalized:
        plan = {
            "source_class": "external_live_lookup",
            "runtime_scope": "",
            "fact_type": fact_type,
            "target_source": "current_user_recent",
            "target": "",
            "missing_context": [],
        }
        return plan

    if "active character's location" in normalized:
        plan = {
            "source_class": "external_live_lookup",
            "runtime_scope": "",
            "fact_type": fact_type,
            "target_source": "active_character_default",
            "target": "",
            "missing_context": [],
        }
        return plan

    target = _target_after_marker(task_body, "explicit location")
    if target:
        plan = {
            "source_class": "external_live_lookup",
            "runtime_scope": "",
            "fact_type": fact_type,
            "target_source": "explicit",
            "target": target,
            "missing_context": [],
        }
        return plan

    target = _target_after_marker(task_body, "explicit target")
    if target:
        plan = {
            "source_class": "external_live_lookup",
            "runtime_scope": "",
            "fact_type": fact_type,
            "target_source": "explicit",
            "target": target,
            "missing_context": [],
        }
        return plan

    if " for " in normalized:
        _, _, target = task_body.rpartition(" for ")
        target = _clean_target(target)
        if target and "location" not in target.lower():
            plan = {
                "source_class": "external_live_lookup",
                "runtime_scope": "",
                "fact_type": fact_type,
                "target_source": "explicit",
                "target": target,
                "missing_context": [],
            }
            return plan

    return None

def _normalize_selector_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM selector payload to the approved plan fields."""
    fact_type = text_or_empty(raw_plan.get("fact_type"))
    if fact_type not in _KNOWN_FACT_TYPES:
        fact_type = "other"

    target_source = text_or_empty(raw_plan.get("target_source"))
    if target_source not in _KNOWN_TARGET_SOURCES:
        target_source = "unknown"

    missing_context = raw_plan.get("missing_context")
    if not isinstance(missing_context, list):
        missing_context = []

    plan = {
        "source_class": "external_live_lookup",
        "runtime_scope": "",
        "fact_type": fact_type,
        "target_source": target_source,
        "target": text_or_empty(raw_plan.get("target")),
        "missing_context": [
            text
            for item in missing_context
            if (text := text_or_empty(item))
        ],
    }
    return plan

_EXTERNAL_LIVE_SELECTOR_PROMPT = '''\
你要为一个 live external fact 请求选择有边界的来源路径。
实时值本身必须来自公开实时/网页证据，绝不能来自 memory。
memory 或近期聊天只能用于解析 target/scope，例如稳定的角色位置或用户近期提到的位置。

# 生成步骤
1. 识别变化中的实时事实类型：weather、temperature、opening_status、
   schedule、price、exchange_rate、current_event_status 或 other。
2. 识别 target_source：
   - explicit: 任务明确给出地点、场馆、市场、产品、事件或公开目标。
   - active_character_default: 任务询问活跃角色的位置/范围。
   - current_user_recent: 任务询问当前用户自己的位置/范围。
   - unknown: 没有可信 target/scope 可见。
3. 只有 target_source 为 explicit 时填写 target。
4. 如果 target 未知，在 missing_context 中写出缺失项，通常是 "location" 或 "target"。

# 输入格式
{
  "task": "Live-context 槽位文本",
  "original_query": "可用时的去上下文化用户问题",
  "current_slot": "槽位标签",
  "known_facts": "之前 RAG2 槽位得到的有序事实"
}

# 输出格式
只返回有效 JSON：
{
  "fact_type": "weather | temperature | opening_status | schedule | price | exchange_rate | current_event_status | other",
  "target_source": "explicit | active_character_default | current_user_recent | unknown",
  "target": "明确目标文本，否则为空字符串",
  "missing_context": ["未解析时缺少的 location 或 target"]
}
'''

_llm_interface = LLInterface()
_external_live_selector_llm = LLInterface()
_external_live_selector_llm_config = LLMCallConfig(
    stage_name=__name__,
    route_name="RAG_SUBAGENT_LLM",
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
    model=RAG_SUBAGENT_LLM_MODEL,
    temperature=0.0,
    top_p=1.0,
    top_k=None,
    max_completion_tokens=RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(
        enabled=RAG_SUBAGENT_LLM_THINKING_ENABLED,
    ),
)

async def _select_external_live_plan(
    task: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Select a bounded external live target path for the requested slot.

    Args:
        task: Slot text supplied by the dispatcher.
        context: RAG2 delegate context.

    Returns:
        Normalized source-selection plan.
    """

    llm_input = project_selector_input_for_llm(task, context)
    system_prompt = SystemMessage(content=_EXTERNAL_LIVE_SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(llm_input, ensure_ascii=False, default=str)
    )
    response = await _external_live_selector_llm.ainvoke(
        [system_prompt, human_message]
    , config=_external_live_selector_llm_config)
    raw_plan = parse_llm_json_output(response.content)
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    plan = _normalize_selector_plan(raw_plan)
    return plan
