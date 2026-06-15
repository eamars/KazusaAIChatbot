"""Person context mode selection."""

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
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
_KNOWN_MODES = {
    "lookup",
    "profile",
    "lookup_profile",
    "user_list",
    "relationship",
    "incompatible",
}

def _strip_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value

def _deterministic_plan(task: str) -> dict[str, Any] | None:
    """Parse structured person-context slots without selector LLM."""
    task_body = _strip_prefix(task)
    normalized = task_body.lower()

    if "unknown speaker" in normalized and ("said" in normalized or '"' in task_body):
        plan = {
            "mode": "incompatible",
            "target": "",
            "reason": "Conversation-evidence",
        }
        return plan

    if "list users" in normalized or "display names" in normalized:
        plan = {
            "mode": "user_list",
            "target": "",
            "reason": "display-name predicate enumeration",
        }
        return plan

    if "relationship" in normalized or "rank users" in normalized:
        plan = {
            "mode": "relationship",
            "target": "",
            "reason": "relationship ranking",
        }
        return plan

    if "active character" in normalized and "profile" in normalized:
        plan = {
            "mode": "profile",
            "target": "active_character",
            "reason": "active character profile",
        }
        return plan

    if "current user" in normalized and "profile" in normalized:
        plan = {
            "mode": "profile",
            "target": "current_user",
            "reason": "current user profile",
        }
        return plan

    if "resolved in slot" in normalized or "speaker found in slot" in normalized:
        plan = {
            "mode": "profile",
            "target": "known_ref",
            "reason": "profile for structured person ref",
        }
        return plan

    if "display name" in normalized and (
        "profile" in normalized or "impression" in normalized
    ):
        plan = {
            "mode": "lookup_profile",
            "target": "display_name",
            "reason": "display-name lookup followed by profile",
        }
        return plan

    if "display name" in normalized or "resolve" in normalized:
        plan = {
            "mode": "lookup",
            "target": "display_name",
            "reason": "display-name identity lookup",
        }
        return plan

    return None

def _normalize_selector_plan(raw_plan: dict[str, Any]) -> dict[str, Any]:
    """Normalize an LLM selector payload to approved fields."""
    mode = text_or_empty(raw_plan.get("mode"))
    if mode not in _KNOWN_MODES:
        mode = "lookup"
    target = text_or_empty(raw_plan.get("target"))
    reason = text_or_empty(raw_plan.get("reason"))
    plan = {
        "mode": mode,
        "target": target,
        "reason": reason,
    }
    return plan

_SELECTOR_PROMPT = '''\
你要为一个 RAG 证据槽位选择一个有边界的 person-context worker 路径。
不要为了未知说话人去搜索聊天历史；这种情况必须先走 Conversation-evidence，
再进入 Person-context。

# 生成步骤
1. 如果任务必须先通过引用消息或内容找到未知说话人，输出
   mode="incompatible"，reason="Conversation-evidence"。
2. display-name 谓词或用户枚举使用 mode="user_list"。
3. 关系排名或关系区间使用 mode="relationship"。
4. 当前用户资料、活跃角色资料，或已由早前槽位解析的人，使用
   mode="profile"。target 设为 current_user、active_character 或 known_ref。
5. display-name -> profile/impression 请求使用 mode="lookup_profile"。
6. 只做身份解析的 display-name 请求使用 mode="lookup"。

# 输入格式
{
  "task": "Person-context 槽位文本",
  "original_query": "可用时的去上下文化用户问题",
  "current_slot": "槽位标签",
  "known_facts": "之前 RAG2 槽位得到的有序事实"
}

# 输出格式
只返回有效 JSON：
{
  "mode": "lookup | profile | lookup_profile | user_list | relationship | incompatible",
  "target": "display_name | current_user | active_character | known_ref | empty",
  "reason": "简短来源选择说明"
}
'''

_llm_interface = LLInterface()
_selector_llm = LLInterface()
_selector_llm_config = LLMCallConfig(
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

async def _select_plan(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Select the bounded person worker path for one slot."""
    deterministic_plan = _deterministic_plan(task)
    if deterministic_plan is not None:
        return deterministic_plan

    llm_input = project_selector_input_for_llm(task, context)
    system_prompt = SystemMessage(content=_SELECTOR_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(llm_input, ensure_ascii=False, default=str)
    )
    response = await _selector_llm.ainvoke([system_prompt, human_message], config=_selector_llm_config)
    raw_plan = parse_llm_json_output(response.content)
    if not isinstance(raw_plan, dict):
        raw_plan = {}
    plan = _normalize_selector_plan(raw_plan)
    return plan
