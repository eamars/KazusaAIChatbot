"""RAG helper agent for scoped current-user continuity evidence."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from openai import OpenAIError

from kazusa_ai_chatbot.config import (

    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
    RAG_SUBAGENT_LLM_MAX_COMPLETION_TOKENS,
    RAG_SUBAGENT_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.db import (
    get_query_text_embedding,
    query_user_memory_units,
    search_user_memory_units_by_keyword,
    search_user_memory_units_by_vector,
)
from kazusa_ai_chatbot.db.schemas import UserMemoryUnitStatus
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.utils import parse_llm_json_output, text_or_empty

from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
logger = logging.getLogger(__name__)

_AGENT_NAME = "user_memory_evidence_agent"
_UNCACHED_REASON = "scoped_user_memory_uncached"
_MAX_ROWS = 5
_MAX_LITERAL_TERMS = 8
_DEFAULT_TRUTH_STATUS = "character_lore_or_interaction_continuity"
_DEFAULT_ORIGIN = "consolidated_interaction"
_DEFAULT_AUTHORITY = "scoped_continuity"


def _cache_status() -> dict[str, Any]:
    """Build the fixed no-cache metadata block for scoped user memory evidence."""
    status = {
        "enabled": False,
        "hit": False,
        "reason": _UNCACHED_REASON,
    }
    return status


def _task_body(task: str) -> str:
    """Remove the top-level capability prefix when present."""
    if ":" not in task:
        stripped_task = task.strip()
        return stripped_task
    _, _, remainder = task.partition(":")
    stripped_task = remainder.strip()
    return stripped_task


def _extract_literal_terms(task: str) -> list[str]:
    """Extract bounded literal anchors for scoped lexical retrieval.

    Args:
        task: Memory-evidence slot text.

    Returns:
        Ordered literal search terms, strongest anchors first.
    """
    task_body = _task_body(task)
    patterns = [
        r'"([^"]+)"',
        r"'([^']+)'",
        r"“([^”]+)”",
        r"‘([^’]+)’",
        r"「([^」]+)」",
        r"『([^』]+)』",
    ]
    literals: list[str] = []
    for pattern in patterns:
        for match in re.findall(pattern, task_body):
            literal = text_or_empty(match).strip()
            if literal and literal not in literals:
                literals.append(literal)
            for cjk_run in re.findall(r"[\u3400-\u9fff]+", literal):
                if len(cjk_run) <= 2:
                    continue
                for index in range(len(cjk_run) - 1):
                    cjk_term = cjk_run[index:index + 2]
                    if cjk_term and cjk_term not in literals:
                        literals.append(cjk_term)
                    if len(literals) >= _MAX_LITERAL_TERMS:
                        return literals

    return literals


def _allows_recency_fallback(task: str) -> bool:
    """Return whether an unanchored request can use bounded recent continuity."""
    task_body = _task_body(task).lower()
    broad_markers = (
        "current user's continuity",
        "current user continuity",
        "private continuity",
        "user memory evidence",
    )
    specific_markers = (
        " about ",
        " around ",
        " relevant to ",
        "exact term",
    )
    has_broad_marker = any(marker in task_body for marker in broad_markers)
    has_specific_marker = any(marker in task_body for marker in specific_markers)
    return_value = has_broad_marker and not has_specific_marker
    return return_value


def _semantic_query_text(task: str, context: dict[str, Any]) -> str:
    """Build the semantic query used for scoped memory vector retrieval.

    Args:
        task: Current memory-evidence slot selected for this worker.
        context: Trusted RAG runtime context containing decontextualized query
            fields from the outer supervisor.

    Returns:
        A compact text payload that preserves slot intent while retaining
        decontextualized query details the initializer may have omitted.
    """

    parts: list[str] = []
    for value in (
        _task_body(task),
        text_or_empty(context.get("current_slot")),
        text_or_empty(context.get("original_query")),
    ):
        if not value:
            continue
        if value in parts:
            continue
        parts.append(value)

    query_text = "\n".join(parts)
    return query_text


def _row_content(row: dict[str, Any]) -> str:
    """Build prompt-facing evidence text from one memory row.

    Args:
        row: User-memory row from storage or a projected evidence payload.

    Returns:
        Fact text plus appraisal and continuity details when they exist. A
        pre-computed ``content`` field is treated as authoritative.
    """

    content = text_or_empty(row.get("content"))
    if content:
        return content

    fact = text_or_empty(row.get("fact"))
    appraisal = text_or_empty(row.get("subjective_appraisal"))
    relationship = text_or_empty(row.get("relationship_signal"))

    parts: list[str] = []
    if fact:
        parts.append(fact)
    if appraisal and appraisal not in parts:
        parts.append(f"Subjective appraisal: {appraisal}")
    if relationship and relationship not in parts:
        parts.append(f"Continuity signal: {relationship}")

    evidence_text = "\n".join(parts)
    return evidence_text


def _project_row(row: dict[str, Any], global_user_id: str) -> dict[str, Any]:
    """Project one raw user-memory row into the public scoped evidence shape."""
    projected_row = dict(row)
    content = _row_content(projected_row)
    projected_row["content"] = content
    projected_row["source_system"] = "user_memory_units"
    projected_row["scope_type"] = "user_continuity"
    projected_row["scope_global_user_id"] = global_user_id

    authority = text_or_empty(projected_row.get("authority")) or _DEFAULT_AUTHORITY
    truth_status = (
        text_or_empty(projected_row.get("truth_status")) or _DEFAULT_TRUTH_STATUS
    )
    origin = text_or_empty(projected_row.get("origin")) or _DEFAULT_ORIGIN
    projected_row["authority"] = authority
    projected_row["truth_status"] = truth_status
    projected_row["origin"] = origin
    return projected_row


def _project_rows(rows: list[dict[str, Any]], global_user_id: str) -> list[dict[str, Any]]:
    """Project bounded raw rows into public scoped evidence rows."""
    projected_rows = [
        _project_row(row, global_user_id)
        for row in rows[:_MAX_ROWS]
        if isinstance(row, dict)
    ]
    return projected_rows


def _append_unique_rows(
    rows: list[dict[str, Any]],
    new_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Append rows until the evidence cap, deduping stable unit ids."""

    merged_rows = list(rows)
    seen_unit_ids = {
        unit_id
        for row in merged_rows
        if isinstance(row, dict)
        if (unit_id := text_or_empty(row.get("unit_id")))
    }
    for row in new_rows:
        if not isinstance(row, dict):
            continue
        unit_id = text_or_empty(row.get("unit_id"))
        if unit_id:
            if unit_id in seen_unit_ids:
                continue
            seen_unit_ids.add(unit_id)
        merged_rows.append(row)
        if len(merged_rows) >= _MAX_ROWS:
            break
    return merged_rows


def _result_payload(
    *,
    rows: list[dict[str, Any]],
    global_user_id: str,
    missing_context: list[str],
    nearby_rows: list[dict[str, Any]] | None = None,
    review: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the public result payload for scoped user-memory evidence."""
    selected_summary = "\n".join(
        content
        for row in rows
        if (content := _row_content(row))
    )
    payload = {
        "selected_summary": selected_summary,
        "memory_rows": rows,
        "source_system": "user_memory_units",
        "scope_type": "user_continuity",
        "scope_global_user_id": global_user_id,
        "missing_context": missing_context,
    }
    if nearby_rows:
        payload["nearby_memory_rows"] = nearby_rows
    if review:
        uncertainty = text_or_empty(review.get("uncertainty"))
        if uncertainty:
            payload["review_uncertainty"] = uncertainty
    return payload


def _agent_result(*, resolved: bool, payload: dict[str, Any]) -> dict[str, Any]:
    """Wrap the scoped user-memory payload in the helper-agent result envelope."""
    result = {
        "resolved": resolved,
        "result": payload,
        "attempts": 1,
        "cache": _cache_status(),
    }
    return result


def _review_candidate_payload(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project scoped memory rows into a compact LLM review payload."""
    candidates: list[dict[str, str]] = []
    for row in rows[:_MAX_ROWS]:
        unit_id = text_or_empty(row.get("unit_id"))
        content = _row_content(row)
        if not unit_id or not content:
            continue
        candidate = {
            "unit_id": unit_id,
            "content": content,
            "unit_type": text_or_empty(row.get("unit_type")),
            "subjective_appraisal": text_or_empty(
                row.get("subjective_appraisal")
            ),
            "relationship_signal": text_or_empty(
                row.get("relationship_signal")
            ),
            "authority": text_or_empty(row.get("authority")),
            "truth_status": text_or_empty(row.get("truth_status")),
            "origin": text_or_empty(row.get("origin")),
        }
        candidates.append(candidate)
    return candidates


def _ordered_valid_ids(raw_value: object, allowed_ids: set[str]) -> list[str]:
    """Keep reviewer-selected ids that refer to retrieved candidate rows."""
    if not isinstance(raw_value, list):
        return_value: list[str] = []
        return return_value

    selected_ids: list[str] = []
    for value in raw_value:
        unit_id = text_or_empty(value)
        if not unit_id or unit_id not in allowed_ids:
            continue
        if unit_id in selected_ids:
            continue
        selected_ids.append(unit_id)
    return selected_ids


def _normalize_review_payload(
    raw_review: object,
    candidates: list[dict[str, str]],
) -> dict[str, Any]:
    """Validate reviewer output without interpreting candidate text."""
    allowed_ids = {
        candidate["unit_id"]
        for candidate in candidates
        if candidate.get("unit_id")
    }
    if not isinstance(raw_review, dict):
        raw_review = {}

    confirmed_ids = _ordered_valid_ids(
        raw_review.get("confirmed_unit_ids"),
        allowed_ids,
    )
    nearby_ids = _ordered_valid_ids(
        raw_review.get("nearby_unit_ids"),
        allowed_ids,
    )
    if not confirmed_ids and not nearby_ids:
        nearby_ids = [
            candidate["unit_id"]
            for candidate in candidates
            if candidate.get("unit_id")
        ]
    review = {
        "confirmed_unit_ids": confirmed_ids,
        "nearby_unit_ids": nearby_ids,
        "summary": text_or_empty(raw_review.get("summary")),
        "uncertainty": text_or_empty(raw_review.get("uncertainty")),
    }
    return review


def _rows_for_review_ids(
    rows: list[dict[str, Any]],
    unit_ids: list[str],
) -> list[dict[str, Any]]:
    """Select retrieved rows by reviewer-approved unit ids."""
    selected_rows: list[dict[str, Any]] = []
    for unit_id in unit_ids:
        for row in rows:
            if text_or_empty(row.get("unit_id")) != unit_id:
                continue
            selected_rows.append(row)
            break
    return selected_rows


_REVIEW_PROMPT = '''\
你审查 scoped current-user memory 候选，用于一个 durable-memory evidence 槽位。
判断哪些检索行直接回答槽位，哪些只是附近上下文，哪些应排除。
不要编造事实，也不要使用 shared/world memory 假设。

# 生成步骤
1. 读取 evidence slot 和 candidate rows。
2. 只有一行直接支持所请求的当前用户私有连续性时，才标为 confirmed。
3. 当某行记住的具体实例命名了满足请求类别的特定 product、plan、decision、
   promise、preference、recommendation 或 prior interaction 时，可视为对更宽槽位的直接支持。
4. `subjective_appraisal` 和 `relationship_signal` 只能作为该行是否回答槽位的辅助上下文，
   不要据此编造额外事实。
5. 相关但不能直接回答槽位的行标为 nearby。
6. 无关行不要放入 confirmed 或 nearby。
7. 如果没有行直接回答槽位，返回空 confirmed ids，并用中文简短说明不确定性。

# 输入格式
{
  "task": "Memory-evidence 槽位文本",
  "candidates": [
    {
      "unit_id": "stable candidate id",
      "content": "memory content",
      "unit_type": "memory category",
      "subjective_appraisal": "optional appraisal",
      "relationship_signal": "optional continuity signal",
      "authority": "source authority",
      "truth_status": "truth status",
      "origin": "memory origin"
    }
  ]
}

# 输出格式
只返回有效 JSON：
{
  "confirmed_unit_ids": ["candidate unit_id"],
  "nearby_unit_ids": ["candidate unit_id"],
  "summary": "已确认行的简短事实摘要，或空字符串",
  "uncertainty": "简短不确定性说明，或空字符串"
}
'''
_llm_interface = LLInterface()
_review_llm = LLInterface()
_review_llm_config = LLMCallConfig(
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


async def _review_user_memory_rows(
    task: str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the reviewer LLM which scoped memory rows answer the slot."""
    candidates = _review_candidate_payload(rows)
    if not candidates:
        review = {
            "confirmed_unit_ids": [],
            "nearby_unit_ids": [],
            "summary": "",
            "uncertainty": "没有检索到可审查的当前用户记忆行。",
        }
        return review

    payload = {
        "task": _task_body(task),
        "candidates": candidates,
    }
    system_prompt = SystemMessage(content=_REVIEW_PROMPT)
    human_message = HumanMessage(
        content=json.dumps(payload, ensure_ascii=False, default=str)
    )
    response = await _review_llm.ainvoke([system_prompt, human_message], config=_review_llm_config)
    raw_review = parse_llm_json_output(response.content)
    review = _normalize_review_payload(raw_review, candidates)
    return review


class UserMemoryEvidenceAgent(BaseRAGHelperAgent):
    """Retrieve current-user continuity evidence from ``user_memory_units`` only."""

    def __init__(self, *, cache_runtime=None) -> None:
        """Create the uncached scoped user-memory evidence agent."""
        super().__init__(
            name=_AGENT_NAME,
            cache_name="",
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 1,
    ) -> dict[str, Any]:
        """Retrieve scoped user continuity for one memory-evidence slot.

        Args:
            task: The ``Memory-evidence:`` slot text.
            context: Runtime context containing the current user scope.
            max_attempts: Ignored. Present for interface parity.

        Returns:
            Standard helper-agent result with scoped user-memory evidence rows.
        """
        del max_attempts

        global_user_id = text_or_empty(context.get("global_user_id"))
        if not global_user_id:
            payload = _result_payload(
                rows=[],
                global_user_id="",
                missing_context=["global_user_id"],
            )
            result = _agent_result(resolved=False, payload=payload)
            return result

        rows: list[dict[str, Any]] = []
        retrieval_mode = ""
        task_body = _task_body(task)
        literal_terms = _extract_literal_terms(task)

        if literal_terms:
            retrieval_mode = "literal"
            for literal in literal_terms:
                keyword_rows = await search_user_memory_units_by_keyword(
                    global_user_id,
                    literal,
                    statuses=[UserMemoryUnitStatus.ACTIVE],
                    limit=_MAX_ROWS,
                )
                if keyword_rows:
                    rows = _append_unique_rows(rows, keyword_rows)
                if len(rows) >= _MAX_ROWS:
                    break

        if not rows and not literal_terms:
            query_embedding: list[float] | None = None
            query_text = _semantic_query_text(task, context)
            try:
                query_embedding = await get_query_text_embedding(query_text)
            except OpenAIError as exc:
                logger.info(
                    f"{_AGENT_NAME} embedding unavailable; "
                    f"falling back to scoped non-vector retrieval: {exc}"
                )

            if query_embedding:
                retrieval_mode = "semantic"
                rows = await search_user_memory_units_by_vector(
                    global_user_id,
                    query_embedding,
                    statuses=[UserMemoryUnitStatus.ACTIVE],
                    limit=_MAX_ROWS,
                )

        if not rows and not literal_terms and _allows_recency_fallback(task):
            retrieval_mode = "recent"
            rows = await query_user_memory_units(
                global_user_id,
                statuses=[UserMemoryUnitStatus.ACTIVE],
                limit=_MAX_ROWS,
            )

        projected_rows = _project_rows(rows, global_user_id)
        nearby_rows: list[dict[str, Any]] = []
        review: dict[str, Any] = {}
        if projected_rows and retrieval_mode == "semantic":
            review_task = _semantic_query_text(task, context)
            review = await _review_user_memory_rows(review_task, projected_rows)
            confirmed_unit_ids = review["confirmed_unit_ids"]
            nearby_unit_ids = review["nearby_unit_ids"]
            nearby_rows = _rows_for_review_ids(projected_rows, nearby_unit_ids)
            projected_rows = _rows_for_review_ids(
                projected_rows,
                confirmed_unit_ids,
            )
        resolved = bool(projected_rows)
        missing_context = [] if resolved else ["user_memory_evidence"]
        payload = _result_payload(
            rows=projected_rows,
            global_user_id=global_user_id,
            missing_context=missing_context,
            nearby_rows=nearby_rows,
            review=review,
        )
        logger.info(
            f"{_AGENT_NAME} output: resolved={resolved} "
            f"scope_global_user_id={global_user_id} "
            f"row_count={len(projected_rows)} "
            f"missing_context={missing_context}"
        )
        result = _agent_result(resolved=resolved, payload=payload)
        return result
