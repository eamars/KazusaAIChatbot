"""Search source subagent for web_agent3."""

from __future__ import annotations

import json
import logging
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage

from kazusa_ai_chatbot.config import (
    SEARXNG_URL,
    WEB_SEARCH_LLM_API_KEY,
    WEB_SEARCH_LLM_BASE_URL,
    WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    WEB_SEARCH_LLM_MODEL,
    WEB_SEARCH_LLM_THINKING_ENABLED,
)
from kazusa_ai_chatbot.llm_interface import (
    LLInterface,
    LLMCallConfig,
    LLMThinkingConfig,
)
import kazusa_ai_chatbot.rag.web_agent3.searxng_tools as searxng_tools
from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision
from kazusa_ai_chatbot.utils import parse_llm_json_output

logger = logging.getLogger(__name__)

SOURCE = "web_search"
SUPPORTED_ACTIONS = ("search",)
DESCRIPTION = '''普通网页搜索。
生成 query 时：
- search: 生成普通网页搜索词。无结果时优先去掉过窄日期、年份和堆叠约束，再改用近义词、英文翻译、精确匹配、site: 限制或排除词。
- 搜索结果 snippet 只是线索，正文读取才是强证据。
- 涉及最新、最近或当前信息时，不要因为 reference_time 自动把当前日期、年份或未来日期写入 search query；只有用户明确要求某一天、某月或某个日期范围时才加入日期。
- 来源核实时，优先搜索任务领域里权威、稳定、可复核的来源路径，不要先使用过窄的时间过滤。
- 当搜索需求同时包含多个独立对象、事实维度或资料类型时，可以保留完整搜索需求；
  来源内部会执行少量聚焦搜索尝试。
'''

_MAX_SEARCH_ATTEMPTS = 4
_SEARCH_QUERY_CHAR_LIMIT = 180
_SEARCH_PURPOSE_CHAR_LIMIT = 220
_ATTEMPT_EVIDENCE_CHAR_LIMIT = 1800
_EXPANDED_SEARCH_OUTPUT_CHAR_LIMIT = 12000
_COMPOSITE_QUERY_WORD_THRESHOLD = 10
_EXPANSION_FALLBACK_NOTE = (
    "Search-attempt expansion produced no valid focused queries; searched the "
    "original dense request as a fallback."
)
_FORBIDDEN_ATTEMPT_FIELD_FRAGMENTS = (
    "schema_version",
    "trace_id",
    "cache_name",
    "cache_hit",
    "cache hit",
    "provider_params",
    "provider params",
    "searxng_params",
    "searxng params",
    "node_id",
    "source_node_id",
    "target_node_id",
    "attempt_index",
    "stage_name",
    "route_name",
    "tool config",
    "tool_config",
    "internal field",
    "internal_field",
)


class _SearchAttempt(TypedDict):
    """One source-local search attempt derived from a dense query."""

    query: str
    purpose: str


def is_enabled() -> bool:
    """Return whether the search source has a configured backend."""
    enabled = bool(SEARXNG_URL)
    return enabled


async def execute(decision: _RouterDecision) -> Any:
    """Execute one web-search decision.

    Args:
        decision: Router decision with the search text in `query`.

    Returns:
        Raw search tool output.
    """
    if decision.action != "search":
        result = {
            "status": "error",
            "source": decision.source,
            "action": decision.action,
            "query": decision.query,
            "message": "web_search supports search actions only.",
        }
        return result

    if not _query_needs_attempt_expansion(decision.query):
        result = await searxng_tools.web_search.ainvoke({"query": decision.query})
        return result

    result = await _execute_expanded_search(decision.query)
    return result


_SEARCH_ATTEMPT_PROMPT = '''\
你是普通网页搜索子代理的一步搜索尝试规划器。

# 任务
把一个过密的网页搜索需求拆成 1 到 4 个可直接用于普通网页搜索的短查询。
每个查询只覆盖一个独立对象、事实维度、资料类型或来源路径。

# 生成规则
1. 保留用户给出的专有名词、版本、型号、组织名、URL 片段和缩写。
2. 删除过宽的比较句式、冗余连接词和不必要的日期限制。
3. 如果原始需求已经是一个简单搜索词，只返回一个等价查询。
4. 不要回答问题，不要引用来源，不要判断事实是否存在。
5. 不要输出搜索引擎参数、内部字段、缓存、trace、schema 或工具配置。
6. 查询应像真实搜索词，而不是一句完整说明。

# 输出格式
只返回合法 JSON：
{
  "attempts": [
    {
      "query": "short focused web search text",
      "purpose": "what evidence this search is trying to find"
    }
  ]
}
'''

_search_attempt_llm = LLInterface()
_search_attempt_llm_config = LLMCallConfig(
    stage_name="web_agent3.web_search.attempt_expansion",
    route_name="WEB_SEARCH_LLM",
    base_url=WEB_SEARCH_LLM_BASE_URL,
    api_key=WEB_SEARCH_LLM_API_KEY,
    model=WEB_SEARCH_LLM_MODEL,
    temperature=0.1,
    top_p=0.8,
    top_k=None,
    max_completion_tokens=WEB_SEARCH_LLM_MAX_COMPLETION_TOKENS,
    presence_penalty=None,
    thinking=LLMThinkingConfig(enabled=WEB_SEARCH_LLM_THINKING_ENABLED),
)


async def _generate_search_attempts(query: str) -> list[dict[str, object]]:
    """Ask the source-local model for focused searches for one dense query.

    Args:
        query: Search request selected by the WebAgent3 router.

    Returns:
        Raw attempt rows from the model. Deterministic code validates and caps
        the returned rows before any search execution.
    """

    human_payload = {"search_request": query}
    try:
        response = await _search_attempt_llm.ainvoke(
            [
                SystemMessage(content=_SEARCH_ATTEMPT_PROMPT),
                HumanMessage(
                    content=json.dumps(human_payload, ensure_ascii=False)
                ),
            ],
            config=_search_attempt_llm_config,
        )
    except Exception as exc:
        logger.exception(f"web_search attempt expansion failed: {exc}")
        return_value: list[dict[str, object]] = []
        return return_value

    try:
        parsed = parse_llm_json_output(response.content)
    except Exception as exc:
        logger.exception(f"web_search attempt expansion parse failed: {exc}")
        return_value = []
        return return_value
    if not isinstance(parsed, dict):
        return_value = []
        return return_value
    raw_attempts = parsed.get("attempts")
    if not isinstance(raw_attempts, list):
        return_value = []
        return return_value

    attempts = [
        attempt
        for attempt in raw_attempts
        if isinstance(attempt, dict)
    ]
    return attempts


def _query_needs_attempt_expansion(query: str) -> bool:
    """Return whether a search string appears to mix independent targets."""

    text = " ".join(str(query or "").strip().split())
    if not text:
        return_value = False
        return return_value
    if "\n" in str(query):
        return_value = True
        return return_value

    lower_text = f" {text.lower()} "
    spaced_signals = (
        " compare ",
        " versus ",
        " vs ",
        " with ",
        " against ",
        " across ",
        " in terms of ",
        " and ",
        " 和 ",
        " 与 ",
    )
    inline_signals = (
        ",",
        ";",
        "/",
        "，",
        "；",
        "、",
    )
    chinese_signals = (
        "比较",
        "对比",
        "以及",
        "和",
        "与",
    )
    signal_count = sum(
        1
        for signal in spaced_signals
        if signal in lower_text
    ) + sum(
        1
        for signal in inline_signals
        if signal in text
    ) + sum(
        1
        for signal in chinese_signals
        if signal in text
    )
    word_count = len(text.split())
    has_comparison_signal = (
        " compare " in lower_text
        or "比较" in text
        or "对比" in text
    )
    needs_expansion = (
        signal_count >= 3
        or (
            signal_count >= 2
            and word_count >= _COMPOSITE_QUERY_WORD_THRESHOLD
        )
        or (
            has_comparison_signal
            and signal_count >= 2
        )
    )
    return_value = needs_expansion
    return return_value


async def _execute_expanded_search(query: str) -> str:
    """Execute bounded source-local search attempts for one dense query."""

    raw_attempts = await _generate_search_attempts(query)
    attempts, planning_notes = _normalize_search_attempts(
        raw_attempts,
        fallback_query=query,
    )
    attempt_results: list[dict[str, str]] = []
    for attempt in attempts:
        result = await searxng_tools.web_search.ainvoke({
            "query": attempt["query"],
        })
        attempt_result = {
            "query": attempt["query"],
            "purpose": attempt["purpose"],
            "status": _search_result_status(result),
            "evidence": _bounded_text(
                result,
                limit=_ATTEMPT_EVIDENCE_CHAR_LIMIT,
            ),
        }
        attempt_results.append(attempt_result)

    rendered_result = _format_expanded_search_result(
        attempt_results,
        planning_notes=planning_notes,
    )
    return rendered_result


def _normalize_search_attempts(
    raw_attempts: list[dict[str, object]],
    *,
    fallback_query: str,
) -> tuple[list[_SearchAttempt], list[str]]:
    """Validate, deduplicate, and cap model-generated search attempts."""

    attempts: list[_SearchAttempt] = []
    seen_queries: set[str] = set()
    for raw_attempt in raw_attempts:
        raw_query = raw_attempt.get("query")
        query = _bounded_attempt_field(
            raw_query,
            limit=_SEARCH_QUERY_CHAR_LIMIT,
        )
        if not query:
            continue
        query_key = query.casefold()
        if query_key in seen_queries:
            continue
        raw_purpose = raw_attempt.get("purpose")
        purpose = _bounded_attempt_field(
            raw_purpose,
            limit=_SEARCH_PURPOSE_CHAR_LIMIT,
        )
        if not purpose:
            purpose = "Search for one focused part of the request."
        attempts.append({"query": query, "purpose": purpose})
        seen_queries.add(query_key)
        if len(attempts) >= _MAX_SEARCH_ATTEMPTS:
            break

    if attempts:
        return attempts, []

    fallback_attempt = {
        "query": _bounded_text(fallback_query, limit=_SEARCH_QUERY_CHAR_LIMIT),
        "purpose": (
            "Search the original request because no narrower attempt was "
            "available."
        ),
    }
    return_value = ([fallback_attempt], [_EXPANSION_FALLBACK_NOTE])
    return return_value


def _bounded_attempt_field(value: object, *, limit: int) -> str:
    """Return one prompt-safe model-generated search-attempt field."""

    if not isinstance(value, str):
        return_value = ""
        return return_value
    text = _bounded_text(value, limit=limit)
    lowered_text = text.casefold()
    normalized_text = lowered_text.replace("_", " ").replace("-", " ")
    for fragment in _FORBIDDEN_ATTEMPT_FIELD_FRAGMENTS:
        normalized_fragment = fragment.casefold().replace("_", " ")
        if fragment.casefold() in lowered_text:
            return_value = ""
            return return_value
        if normalized_fragment in normalized_text:
            return_value = ""
            return return_value
    return text


def _bounded_text(value: object, *, limit: int) -> str:
    """Return stripped text clipped to the given character limit."""

    text = str(value or "").strip()
    if len(text) <= limit:
        return text

    clipped_text = text[:limit].rstrip()
    return_value = f"{clipped_text}..."
    return return_value


def _search_result_status(result: str) -> str:
    """Classify one search attempt result without judging factual truth."""

    stripped_result = result.strip()
    if stripped_result.startswith("Error:"):
        return_value = "tool error"
        return return_value
    if stripped_result == "No results found.":
        return_value = "no useful result"
        return return_value
    return_value = "source evidence returned"
    return return_value


def _format_expanded_search_result(
    attempt_results: list[dict[str, str]],
    *,
    planning_notes: list[str] | tuple[str, ...] = (),
) -> str:
    """Render expanded source results as prompt-safe semantic evidence."""

    lines = ["Search attempts:"]
    weak_rows: list[str] = [
        note
        for note in planning_notes
        if isinstance(note, str) and note.strip()
    ]
    for index, attempt_result in enumerate(attempt_results, start=1):
        lines.append(f"{index}. Query: {attempt_result['query']}")
        lines.append(f"   Purpose: {attempt_result['purpose']}")
        lines.append(f"   Result: {attempt_result['status']}")
        lines.append("   Key evidence:")
        evidence_lines = attempt_result["evidence"].splitlines()
        if evidence_lines:
            for evidence_line in evidence_lines:
                if evidence_line.strip():
                    lines.append(f"   {evidence_line}")
        else:
            lines.append("   No evidence text returned.")
        if attempt_result["status"] != "source evidence returned":
            weak_rows.append(
                f"No useful source evidence returned for query: "
                f"{attempt_result['query']}"
            )

    lines.append("")
    lines.append("Missing or weak coverage:")
    if weak_rows:
        for weak_row in weak_rows:
            lines.append(f"- {weak_row}")
    else:
        lines.append(
            "- No tool-level missing result was observed; downstream "
            "evaluation still needs to judge source relevance and freshness."
        )

    lines.append("")
    lines.append("Recommended narrower search focus:")
    if weak_rows:
        lines.append(
            "- Retry missing targets with fewer constraints or a more "
            "source-oriented query."
        )
    else:
        lines.append(
            "- Read the most relevant returned URLs before treating snippets "
            "as strong evidence."
        )

    rendered_result = "\n".join(lines)
    bounded_result = _bounded_text(
        rendered_result,
        limit=_EXPANDED_SEARCH_OUTPUT_CHAR_LIMIT,
    )
    return bounded_result
