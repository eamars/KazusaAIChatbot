"""Conversation evidence worker selection."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (

    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.rag.conversation_evidence.contracts import (
    _AGENT_NAME,
    _ConversationProjection,
)
from kazusa_ai_chatbot.rag.evidence_coverage import (
    requested_coverage_items,
    task_requires_value_evidence,
)
from kazusa_ai_chatbot.rag.hybrid_retrieval import candidate_prompt_text
from kazusa_ai_chatbot.rag.prompt_projection import project_selector_input_for_llm
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

_COUNT_INTENT_RE = re.compile(
    r"\b(?:count|counts|counted|counting)\b",
    flags=re.IGNORECASE,
)

_EXPLICIT_SPEAKER_SCOPE_PATTERN = re.compile(
    r"\bspeaker\s*=\s*(current_user|active_character|any_speaker)\b",
    re.IGNORECASE,
)

_PERSON_SPEAKER_SCOPE_PATTERN = re.compile(
    r"\bspeaker\s*=\s*person\s+resolved\s+in\s+slot\s+\d+\b",
    re.IGNORECASE,
)

_SCOPE_CURRENT_USER = "current_user"

_SCOPE_ACTIVE_CHARACTER = "active_character"

_SCOPE_ANY_SPEAKER = "any_speaker"

_SCOPE_PERSON_RESOLVED = "person_resolved"

_KNOWN_WORKERS = {
    "conversation_search_agent",
    "conversation_filter_agent",
    "conversation_aggregate_agent",
    "incompatible",
}

_RECALL_OWNED_TASK_MARKERS = (
    "active agreement",
    "active promise",
    "active commitment",
    "ongoing agreement",
    "ongoing promise",
    "ongoing commitment",
    "current plan",
    "current plans",
    "open loop",
    "open loops",
    "unresolved loop",
    "unresolved loops",
    "next step",
    "episode position",
    "episode state",
    "current episode state",
    "where the current episode left off",
    "where current episode left off",
)

_RETRIEVAL_CONFIRMATION_MARKERS = (
    "retrieve messages",
    "retrieve recent messages",
    "find context",
    "messages around",
    "messages near",
    "context mentioning",
    "mentioning",
    "containing",
)

_VALUE_IDENTIFICATION_MARKERS = (
    "identify",
    "who said",
    "which speaker",
    "which user",
    "which person",
    "what time",
    "when was",
    "when did",
    "count ",
    "how many",
    "ranking",
    "grouped",
    "aggregate",
)

_EXACT_ANCHOR_CHARS = (
    '"',
    "'",
    "\u2018",
    "\u2019",
    "\u201c",
    "\u201d",
    "\u300c",
    "\u300d",
    "\u300e",
    "\u300f",
)

def _strip_prefix(task: str) -> str:
    """Remove the semantic capability prefix when present."""
    if ":" not in task:
        return task.strip()
    _, _, remainder = task.partition(":")
    return_value = remainder.strip()
    return return_value

def _speaker_scope(task: str) -> str:
    """Extract the optional conversation author scope from a slot.

    Args:
        task: Conversation-evidence slot text.

    Returns:
        One approved scope constant, or an empty string for legacy unscoped
        slots.
    """

    task_body = _strip_prefix(task)
    if _PERSON_SPEAKER_SCOPE_PATTERN.search(task_body):
        return_value = _SCOPE_PERSON_RESOLVED
        return return_value

    explicit_match = _EXPLICIT_SPEAKER_SCOPE_PATTERN.search(task_body)
    if explicit_match is None:
        return_value = ""
        return return_value

    return_value = explicit_match.group(1).lower()
    return return_value

def _requires_person_ref(task: str) -> bool:
    """Return whether a slot needs a prior structured person reference.

    Args:
        task: Conversation-evidence slot text.

    Returns:
        True when the slot explicitly depends on a person from an earlier slot.
    """

    if _speaker_scope(task) == _SCOPE_PERSON_RESOLVED:
        return True

    normalized = _strip_prefix(task).lower()
    return_value = "resolved in slot" in normalized
    return return_value

def _recall_owned_task_reason(task_body: str) -> str:
    """Return Recall when a conversation task asks for episode state.

    Args:
        task_body: Conversation-evidence slot text without the capability
            prefix.

    Returns:
        ``"Recall"`` when the task belongs to active episode recall; otherwise
        an empty string.
    """

    normalized = task_body.lower()
    is_recall_task = any(
        marker in normalized
        for marker in _RECALL_OWNED_TASK_MARKERS
    )
    reason = "Recall" if is_recall_task else ""
    return reason

def _deterministic_plan(task: str) -> dict[str, Any] | None:
    """Parse structured conversation-evidence slots without selector LLM."""

    task_body = _strip_prefix(task)
    normalized = task_body.lower()
    speaker_scope = _speaker_scope(task)
    requires_person_ref = _requires_person_ref(task)
    incompatible_reason = _recall_owned_task_reason(task_body)

    if incompatible_reason:
        plan = {
            "worker": "incompatible",
            "reason": incompatible_reason,
            "requires_person_ref": False,
        }
        return plan

    if "durable" in normalized or "official address" in normalized:
        plan = {
            "worker": "incompatible",
            "reason": "Memory-evidence",
            "requires_person_ref": False,
        }
        return plan

    if (
        _COUNT_INTENT_RE.search(normalized)
        or "how many" in normalized
        or "ranking" in normalized
        or "grouped" in normalized
        or "aggregate" in normalized
    ):
        plan = {
            "worker": "conversation_aggregate_agent",
            "reason": "aggregate/count conversation evidence",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        any(anchor in task_body for anchor in _EXACT_ANCHOR_CHARS)
        or "exact phrase" in normalized
        or "exact term" in normalized
        or "url" in normalized
        or "filename" in normalized
        or "literal" in normalized
    ):
        plan = {
            "worker": "conversation_search_agent",
            "reason": "hybrid literal phrase, URL, filename, or exact anchor",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        "relation=" in normalized
        or "attachment" in normalized
        or "screenshot" in normalized
        or "image" in normalized
        or "preceding" in normalized
        or "previous" in normalized
    ):
        plan = {
            "worker": "conversation_search_agent",
            "reason": "relation or media-bearing conversation evidence",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        "topic" in normalized
        or "about " in normalized
        or "semantic" in normalized
    ):
        plan = {
            "worker": "conversation_search_agent",
            "reason": "semantic or fuzzy topic conversation evidence",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    if (
        "resolved in slot" in normalized
        or "recent messages from" in normalized
        or (
            speaker_scope
            and "recent messages" in normalized
        )
        or "date-window" in normalized
        or "from timestamp" in normalized
        or "to timestamp" in normalized
    ):
        plan = {
            "worker": "conversation_filter_agent",
            "reason": "structured conversation filter",
            "requires_person_ref": requires_person_ref,
        }
        return plan

    return None

def _normalize_selector_plan(
    raw_plan: dict[str, Any],
    task: str,
) -> dict[str, Any]:
    """Normalize selector output while keeping dependency checks deterministic.

    Args:
        raw_plan: Parsed selector LLM output.
        task: Conversation-evidence slot text being planned.

    Returns:
        Approved worker plan with person-reference requirements derived from
        the slot text instead of the selector payload.
    """
    worker = text_or_empty(raw_plan.get("worker"))
    if worker == "conversation_keyword_agent":
        worker = "conversation_search_agent"
    if worker not in _KNOWN_WORKERS:
        worker = "conversation_search_agent"
    reason = text_or_empty(raw_plan.get("reason"))
    requires_person_ref = worker != "incompatible" and _requires_person_ref(task)
    plan = {
        "worker": worker,
        "reason": reason,
        "requires_person_ref": requires_person_ref,
    }
    return plan

_SELECTOR_PROMPT = '''\
你要为一个 RAG 证据槽位选择一个有边界的 conversation-history worker 路径。
不要用聊天历史路径回答 durable memory、活跃 episode 进度、用户资料或网页内容。

# 生成步骤
1. 如果任务询问活跃/当前约定或 episode 状态，输出
   worker="incompatible"，reason="Recall"。
2. 如果任务询问 durable world fact，输出 worker="incompatible"，
   reason="Memory-evidence"。
3. 模糊话题、语义消息证据、精确短语、URLs、filenames、字面词和引用消息来源，
   使用 conversation_search_agent。该 worker 执行语义加字面锚点的混合检索。
4. 已知用户、时间或日期窗口检索使用 conversation_filter_agent。
5. 计数、排名或分组统计使用 conversation_aggregate_agent。
6. 不要判断是否需要结构化人物引用；该依赖由槽位文本的确定性校验负责。

# 输入格式
{
  "task": "Conversation-evidence 槽位文本",
  "original_query": "可用时的去上下文化用户问题",
  "current_slot": "槽位标签",
  "known_facts": "之前 RAG2 槽位得到的有序事实"
}

# 输出格式
只返回有效 JSON：
{
  "worker": "conversation_search_agent | conversation_filter_agent | conversation_aggregate_agent | incompatible",
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
    """Select the bounded conversation worker path for one slot."""
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
    plan = _normalize_selector_plan(raw_plan, task)
    return plan

def _iter_known_refs(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect structured refs from previous RAG2 known facts."""
    refs: list[dict[str, Any]] = []
    known_facts = context.get("known_facts")
    if not isinstance(known_facts, list):
        return refs

    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        raw_result = fact.get("raw_result")
        if not isinstance(raw_result, dict):
            continue
        raw_refs = raw_result.get("resolved_refs")
        if not isinstance(raw_refs, list):
            continue
        for ref in raw_refs:
            if isinstance(ref, dict):
                refs.append(ref)
    return refs

def _coverage_requires_value_evidence(task: str, context: dict[str, Any]) -> bool:
    """Preserve value-evidence intent across narrowed continuation slots."""

    if task_requires_value_evidence(task):
        return_value = True
        return return_value

    current_items = requested_coverage_items(task)
    if not current_items:
        return_value = False
        return return_value

    known_facts = context.get("known_facts")
    if not isinstance(known_facts, list):
        return_value = False
        return return_value

    for fact in known_facts:
        if not isinstance(fact, dict):
            continue
        if _known_fact_carries_value_intent(fact, current_items):
            return_value = True
            return return_value

    return_value = False
    return return_value

def _known_fact_carries_value_intent(
    fact: dict[str, Any],
    current_items: list[str],
) -> bool:
    """Return whether one prior fact preserves value intent for current items."""

    if fact.get("agent") != _AGENT_NAME:
        return_value = False
        return return_value
    if bool(fact.get("resolved")):
        return_value = False
        return return_value

    slot = text_or_empty(fact.get("slot"))
    if not task_requires_value_evidence(slot):
        return_value = False
        return return_value

    prior_items = _prior_requested_items(fact, slot)
    if not prior_items:
        return_value = False
        return return_value

    current_item_set = set(current_items)
    prior_item_set = set(prior_items)
    return_value = bool(current_item_set & prior_item_set)
    return return_value

def _prior_requested_items(fact: dict[str, Any], slot: str) -> list[str]:
    """Read requested coverage items from a prior fact or its slot text."""

    raw_result = fact.get("raw_result")
    if isinstance(raw_result, dict):
        coverage = raw_result.get("coverage")
        if isinstance(coverage, dict):
            requested_items = coverage.get("requested_items")
            if isinstance(requested_items, list):
                items = [
                    item
                    for value in requested_items
                    if (item := text_or_empty(value))
                ]
                return items

    items = requested_coverage_items(slot)
    return items

def _first_person_ref(context: dict[str, Any]) -> dict[str, Any] | None:
    """Return the first structured person reference from previous slots."""
    for ref in _iter_known_refs(context):
        if ref.get("ref_type") == "person":
            return ref
    return_value = None
    return return_value

def _worker_context(
    context: dict[str, Any],
    person_ref: dict[str, Any] | None,
    speaker_scope: str,
) -> dict[str, Any]:
    """Build worker context with approved structured handoff values.

    Args:
        context: Runtime context from the RAG supervisor.
        person_ref: Prior resolved person reference, when the slot depends on
            one.
        speaker_scope: Optional author scope declared by the conversation slot.

    Returns:
        Worker context with only the user filter implied by the author scope.
    """
    worker_context = dict(context)
    worker_context["exclude_current_question"] = True
    if speaker_scope:
        worker_context["conversation_user_scope"] = speaker_scope
    else:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)

    if speaker_scope == _SCOPE_ANY_SPEAKER:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)

    if speaker_scope == _SCOPE_ACTIVE_CHARACTER:
        worker_context.pop("global_user_id", None)
        worker_context.pop("display_name", None)
        character_profile = context.get("character_profile")
        resolved_global_user_id = ""
        resolved_display_name = ""
        if isinstance(character_profile, dict):
            resolved_global_user_id = text_or_empty(
                character_profile.get("global_user_id")
            )
            resolved_display_name = text_or_empty(character_profile.get("name"))
            if resolved_global_user_id:
                worker_context["global_user_id"] = resolved_global_user_id
            if resolved_display_name:
                worker_context["display_name"] = resolved_display_name
        if not resolved_global_user_id or not resolved_display_name:
            platform_channel_id = text_or_empty(context.get("platform_channel_id"))
            logger.warning(
                "conversation_evidence: speaker=active_character requested "
                "without character_profile "
                f"platform_channel_id={platform_channel_id}"
            )
        else:
            logger.debug(
                "conversation_evidence active-character scope resolved: "
                f"global_user_id={resolved_global_user_id}"
            )

    if person_ref is not None:
        global_user_id = text_or_empty(person_ref.get("global_user_id"))
        display_name = text_or_empty(person_ref.get("display_name"))
        if global_user_id:
            worker_context["global_user_id"] = global_user_id
        if display_name:
            worker_context["display_name"] = display_name
    return worker_context
