"""LLM review for recall candidates."""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (

    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.rag.recall.contracts import (
    _ACTIVE_MODES,
    _CONFLICT_LIMIT,
    _SOURCE_ORDERS,
)
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

def _has_source_conflict(candidates: list[dict[str, str]]) -> bool:
    """Detect source-level disagreement before using transcript proof."""

    source_claims: dict[str, str] = {}
    for candidate in candidates:
        source = candidate["source"]
        if source == "conversation_history" or source in source_claims:
            continue
        source_claims[source] = candidate["claim"].casefold()

    if len(source_claims) <= 1:
        return False

    distinct_claims = set(source_claims.values())
    return_value = len(distinct_claims) > 1
    return return_value

def _conflict_notes(candidates: list[dict[str, str]]) -> list[str]:
    """Create compact conflict notes from candidate source disagreement."""

    if not _has_source_conflict(candidates):
        return_value: list[str] = []
        return return_value

    notes: list[str] = []
    seen_sources: set[str] = set()
    for candidate in candidates:
        source = candidate["source"]
        if source in seen_sources or source == "conversation_history":
            continue
        seen_sources.add(source)
        notes.append(f"{source}: {candidate['claim']}")
        if len(notes) >= _CONFLICT_LIMIT:
            break
    return notes

def _rank_candidates(mode: str, candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Sort candidates by the fixed source authority order for a Recall mode."""

    source_order = _SOURCE_ORDERS[mode]
    order_index = {
        source: index
        for index, source in enumerate(source_order)
    }
    ranked_candidates = sorted(
        candidates,
        key=lambda candidate: (
            order_index.get(candidate["source"], len(source_order)),
            0 if candidate["lifecycle_status"] in {"active", "pending"} else 1,
            candidate["evidence_time"],
        ),
    )
    return ranked_candidates

def _recall_type_for(mode: str, selected: dict[str, str], conflicts: list[str]) -> str:
    """Map slot mode and selected source to the public Recall result type."""

    if mode == "exact_agreement_history":
        return "exact_history"
    if selected["source"] == "user_memory_units":
        return "durable_commitment"
    if conflicts:
        return "mixed"
    return_value = "active_episode_agreement"
    return return_value

def _freshness_basis(selected: dict[str, str], mode: str, progress_unavailable: bool) -> str:
    """Explain why the selected evidence source is authoritative."""

    if progress_unavailable and selected["source"] == "user_memory_units":
        return_value = (
            "Active-episode state was unavailable, so active commitment memory "
            "is the best durable ongoing source."
        )
        return return_value
    evidence_time = selected["evidence_time"]
    if evidence_time:
        return_value = (
            f"Selected {selected['source']} for {mode}; evidence_time={evidence_time}."
        )
        return return_value
    return_value = f"Selected {selected['source']} for {mode}."
    return return_value

def _candidate_review_payload(
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    """Project recall candidates into a compact LLM review payload."""

    payload: list[dict[str, str]] = []
    for index, candidate in enumerate(candidates):
        payload.append(
            {
                "index": str(index),
                "source": candidate["source"],
                "claim": candidate["claim"],
                "temporal_scope": candidate["temporal_scope"],
                "lifecycle_status": candidate["lifecycle_status"],
                "evidence_time": candidate["evidence_time"],
                "authority": candidate["authority"],
            }
        )
    return payload

def _ordered_valid_candidate_indexes(
    raw_value: object,
    max_index: int,
) -> list[int]:
    """Keep reviewer-selected indexes that refer to retrieved candidates."""

    if not isinstance(raw_value, list):
        indexes: list[int] = []
        return indexes

    selected_indexes: list[int] = []
    for value in raw_value:
        if isinstance(value, int):
            index = value
        else:
            value_text = text_or_empty(value)
            if not value_text.isdigit():
                continue
            index = int(value_text)
        if index < 0 or index >= max_index:
            continue
        if index in selected_indexes:
            continue
        selected_indexes.append(index)
    return selected_indexes

def _normalize_recall_review(
    raw_review: object,
    candidate_count: int,
) -> dict[str, Any]:
    """Validate recall reviewer output without interpreting candidate text."""

    if not isinstance(raw_review, dict):
        raw_review = {}

    confirmed_indexes = _ordered_valid_candidate_indexes(
        raw_review.get("confirmed_candidate_indexes"),
        candidate_count,
    )
    nearby_indexes = _ordered_valid_candidate_indexes(
        raw_review.get("nearby_candidate_indexes"),
        candidate_count,
    )
    source_hints = []
    raw_source_hints = raw_review.get("source_hints")
    if isinstance(raw_source_hints, list):
        for item in raw_source_hints[:3]:
            source_hint = text_or_empty(item)
            if source_hint:
                source_hints.append(source_hint)

    review = {
        "confirmed_candidate_indexes": confirmed_indexes,
        "nearby_candidate_indexes": nearby_indexes,
        "summary": text_or_empty(raw_review.get("summary")),
        "uncertainty": text_or_empty(raw_review.get("uncertainty")),
        "source_hints": source_hints,
    }
    return review

def _recall_slot_needs_candidate_review(
    *,
    task: str,
    mode: str,
    progress_unavailable: bool,
    candidates: list[dict[str, str]],
) -> bool:
    """Return whether fallback recall candidates need semantic arbitration."""

    if not progress_unavailable:
        return_value = False
        return return_value
    if not candidates:
        return_value = False
        return return_value

    durable_candidate_count = sum(
        1
        for candidate in candidates
        if candidate["source"] == "user_memory_units"
    )
    if durable_candidate_count <= 1:
        return_value = False
        return return_value
    if mode in _ACTIVE_MODES:
        return_value = True
        return return_value

    task_text = task.casefold()
    review_markers = (
        "relevant to",
        "about",
        "regarding",
        "for ",
    )
    needs_review = any(marker in task_text for marker in review_markers)
    return_value = needs_review
    return return_value

def _review_observation_candidates(
    candidates: list[dict[str, str]],
    indexes: list[int] | None = None,
) -> list[dict[str, str]]:
    """Expose unconfirmed recall candidates as continuation observations."""

    selected_candidates = candidates[:6]
    if indexes is not None:
        selected_candidates = [
            candidates[index]
            for index in indexes
            if 0 <= index < len(candidates)
        ][:6]

    observation_candidates: list[dict[str, str]] = []
    for candidate in selected_candidates:
        observation_candidates.append(
            {
                "content": candidate["claim"],
                "source": f"recall:{candidate['source']}",
                "summary": candidate["claim"],
            }
        )
    return observation_candidates

def _review_source_hints(review: dict[str, Any]) -> list[dict[str, str]]:
    """Build continuation source hints from recall-review uncertainty."""

    source_hints: list[dict[str, str]] = []
    for hint in review["source_hints"]:
        source_hints.append(
            {
                "kind": "recall",
                "source": hint,
            }
        )
    if not source_hints:
        source_hints.append(
            {
                "kind": "recall",
                "source": (
                    "Fallback recall candidates did not directly answer; "
                    "conversation or memory evidence may be needed for exact "
                    "historical details."
                ),
            }
        )
    return source_hints

_RECALL_REVIEW_PROMPT = '''\
你审查 Recall candidates，用于一个 active agreement、commitment、plan 或
episode-position 槽位。判断是否有候选项直接回答该槽位。不要编造事实。

# 生成步骤
1. 读取 Recall slot，识别必要请求细节：具体实体、日期/时间、地点、状态、
   obligations、人物和事件上下文。
2. 只有候选项是槽位请求的同一个具体 agreement、promise、plan 或 episode state，
   才确认该候选项。
3. 如果槽位包含多个必要细节，confirmed candidate 必须覆盖所有细节，或清楚蕴含所有细节。
   只匹配一个细节不是直接支持。
4. 对 `active_episode_agreement` 要严格：durable commitment candidate 必须描述同一个
   当前/近期约定，而不是同一宽泛话题下的另一个 active commitment。
5. 对 `active_episode_agreement`，如果 milestone、objective fact 或 status record
   直接说明所请求约定的履行、取消、付款、完成或当前状态，可以确认槽位。
6. 不要仅因候选项和槽位同属甜点、硬件、会议或任务等宽泛类别就确认。
7. 相关但不足以回答的候选项标为 nearby。
8. 不确定时保持 unconfirmed，让调用方搜索 conversation 或 memory evidence 获取精确历史细节。
9. 如果没有候选项直接回答，返回空 confirmed indexes，并用中文给出下一步可能需要的证据来源提示。

# 输入格式
{
  "task": "Recall 槽位文本",
  "mode": "active_episode_agreement | durable_commitment | episode_position | exact_agreement_history",
  "candidates": [
    {
      "index": "0",
      "source": "source name",
      "claim": "candidate claim",
      "temporal_scope": "candidate time scope",
      "lifecycle_status": "active | pending | historical",
      "evidence_time": "visible timestamp if available",
      "authority": "source authority label"
    }
  ]
}

# 输出格式
只返回有效 JSON：
{
  "confirmed_candidate_indexes": [0],
  "nearby_candidate_indexes": [1],
  "summary": "已确认候选项的简短事实摘要，或空字符串",
  "uncertainty": "简短不确定性说明，或空字符串",
  "source_hints": ["没有候选项确认槽位时的简短提示"]
}
'''

_llm_interface = LLInterface()
_recall_review_llm = LLInterface()
_recall_review_llm_config = LLMCallConfig(
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

async def _review_recall_candidates(
    *,
    task: str,
    mode: str,
    candidates: list[dict[str, str]],
) -> dict[str, Any]:
    """Ask the reviewer LLM whether recall candidates answer the slot."""

    if not candidates:
        review = {
            "confirmed_candidate_indexes": [],
            "nearby_candidate_indexes": [],
            "summary": "",
            "uncertainty": "没有可用的 Recall 候选项。",
            "source_hints": [],
        }
        return review

    payload = {
        "task": task,
        "mode": mode,
        "candidates": _candidate_review_payload(candidates),
    }
    system_prompt = SystemMessage(content=_RECALL_REVIEW_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(payload, ensure_ascii=False, default=str)
    )
    response = await _recall_review_llm.ainvoke([system_prompt, human_message], config=_recall_review_llm_config)
    raw_review = parse_llm_json_output(response.content)
    review = _normalize_recall_review(raw_review, len(candidates))
    logger.info(
        f"recall_agent candidate review: "
        f"confirmed={review['confirmed_candidate_indexes']} "
        f"nearby={review['nearby_candidate_indexes']} "
        f"uncertainty={review['uncertainty']!r}"
    )
    logger.debug(f"recall_agent candidate review raw={raw_review!r}")
    return review
