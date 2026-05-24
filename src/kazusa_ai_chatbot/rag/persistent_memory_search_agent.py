"""RAG helper agent: semantic persistent-memory search."""

from __future__ import annotations

import json
import logging
import re
from string import Template
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import (
    CHARACTER_GLOBAL_USER_ID,
    RAG_HYBRID_LITERAL_ANCHOR_LIMIT,
    RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
    RAG_SEARCH_DEFAULT_TOP_K,
    RAG_SEARCH_MAX_TOP_K,
    RAG_SEARCH_SELECTED_LIMIT,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.rag.memory_retrieval_tools import search_persistent_memory
from kazusa_ai_chatbot.rag.memory_retrieval_tools import (
    search_persistent_memory_keyword,
)
from kazusa_ai_chatbot.rag.cache2_policy import (
    PERSISTENT_MEMORY_SEARCH_CACHE_NAME,
    build_persistent_memory_search_cache_key,
    build_persistent_memory_search_dependencies,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.hybrid_retrieval import (
    HybridCandidate,
    merge_hybrid_candidates,
)
from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
    project_tool_result_for_llm,
)
from kazusa_ai_chatbot.rag.search_runtime import (
    apply_source_memory_runtime_constraints,
    literal_anchors_from_text,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_GENERATOR_PROMPT = Template("""\
You generate retrieval parameters only for `search_persistent_memory`.

# Scope
- Produce parameters only for `search_persistent_memory`.
- `search_query` must be a natural-language memory query, not a keyword list.
- `literal_anchors` is optional. Include only proper nouns, technical terms,
  short phrases, URLs, filenames, or quoted text that must match literally.
  Do not split a full sentence into a word list.
- `search_query` should target evidence needed by the original question. Do
  not assume the memory category is fact, impression, opinion, or one specific
  source unless the task says so.
- `source_global_user_id` is a privacy boundary, not a relevance hint. Omit it
  by default. Fill it only when the task explicitly asks for memory triggered,
  provided, or promised by one source user.
- If `feedback` says filters are too narrow, query is too abstract, or no
  relevant memory was found, rewrite the next attempt.
- Use English for generated control/status text. Preserve literal anchors,
  names, quotes, URLs, filenames, and source text in their original language.

# Field Constraints
- `source_global_user_id`: fill only when the task explicitly requests a
  source-user filter and context or known_facts contains a clear UUID source
  user id. Platform channel id, username, nickname, current speaker, and target
  person are not valid source-user filters.

# Generation Procedure
1. Read `task` and identify the durable-memory evidence needed.
2. Read `context` and `known_facts`; use only explicit filters.
3. Omit `source_global_user_id` unless the task explicitly requires a memory
   source-user filter.
4. If feedback says the filter is too narrow or the query is too abstract,
   rewrite the query and relax unnecessary filters.
5. Output natural-language `search_query` plus necessary filter fields.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "context": "known facts and runtime hints",
  "feedback": "previous judge feedback, or empty string"
}

# Output Format
Return valid JSON only:
{
  "search_query": "string",
  "literal_anchors": ["string, optional; at most 5 anchors"],
  "top_k": $default_top_k,
  "source_global_user_id": "UUID string or omitted"
}
""").substitute(default_top_k=RAG_SEARCH_DEFAULT_TOP_K)
_generator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

_JUDGE_PROMPT = """\
You judge whether a `search_persistent_memory` result resolves the current slot.

# Task
- Decide whether the current result is enough to resolve the slot.
- If unresolved, feedback must be specific and executable.

# Audit Procedure
1. Read `task` and identify the memory evidence the slot needs.
2. Inspect `result` for directly relevant durable memories, errors, or empty
   output.
3. Return `resolved: true` only when the result is enough to resolve the slot.
4. If unresolved, state whether the next attempt should rewrite the query,
   relax filters, or change topic angle.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "result": "tool result from search_persistent_memory"
}

# Common Feedback Directions
- Query too abstract; rewrite around concrete evidence required by the
  original question.
- Filters too narrow; remove source filters.
- Returned memories irrelevant; change topic angle.
- No relevant memory; relax filters or rewrite query.
- Use English for generated control/status text while preserving source text.

# Output Format
Return valid JSON only:
{
  "resolved": true or false,
  "feedback": "string"
}
"""
_judge_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)


def _normalize_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Keep only valid `search_persistent_memory` arguments.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``search_persistent_memory``.
    """
    args: dict[str, Any] = {}

    search_query = text_or_empty(raw_args.get("search_query"))
    if search_query:
        args["search_query"] = search_query

    literal_anchors = _normalize_literal_anchors(raw_args.get("literal_anchors"))
    if literal_anchors:
        args["literal_anchors"] = literal_anchors

    top_k = raw_args.get("top_k", RAG_SEARCH_DEFAULT_TOP_K)
    if isinstance(top_k, int) and not isinstance(top_k, bool) and top_k > 0:
        args["top_k"] = min(top_k, RAG_SEARCH_MAX_TOP_K)
    else:
        args["top_k"] = RAG_SEARCH_DEFAULT_TOP_K

    for key in ("source_global_user_id",):
        raw_val = raw_args.get(key)
        if raw_val is None:
            continue
        value = text_or_empty(raw_val)
        if value:
            args[key] = value

    _erase_character_source_global_user_id(args)
    return args


def _normalize_literal_anchors(value: object) -> list[str]:
    """Normalize optional literal anchors from the generator output."""

    if not isinstance(value, list):
        return_value: list[str] = []
        return return_value

    anchors: list[str] = []
    for item in value:
        anchor = text_or_empty(item)
        if not anchor:
            continue
        if anchor not in anchors:
            anchors.append(anchor)
        if len(anchors) >= RAG_HYBRID_LITERAL_ANCHOR_LIMIT:
            break

    return anchors


def _erase_character_source_global_user_id(args: dict[str, Any]) -> None:
    """Erase the character ID when it is used as a memory source filter.

    Args:
        args: Mutable normalized subagent arguments.
    """
    source_global_user_id = args.get("source_global_user_id")
    if source_global_user_id == CHARACTER_GLOBAL_USER_ID:
        args.pop("source_global_user_id")


def _slot_number(task: str) -> int | None:
    """Extract a referenced slot number from a task description.

    Args:
        task: Slot description that may contain "slot N".

    Returns:
        Integer slot number, or None when no reference is present.
    """
    match = re.search(r"slot\s+(\d+)", task, flags=re.IGNORECASE)
    if match is None:
        return None
    return_value = int(match.group(1))
    return return_value


def _resolved_user_from_known_facts(task: str, context: dict[str, Any]) -> dict[str, Any]:
    """Find the user resolved by the slot referenced in the memory-search task.

    Args:
        task: Slot description produced by the outer-loop supervisor.
        context: Agent context containing known_facts.

    Returns:
        Dict with known resolved-user fields, or an empty dict when unavailable.
    """
    slot_number = _slot_number(task)
    known_facts = context.get("known_facts", [])
    if slot_number is None or not isinstance(known_facts, list):
        return_value = {}
        return return_value
    if slot_number < 1 or slot_number > len(known_facts):
        return_value = {}
        return return_value

    fact = known_facts[slot_number - 1]
    if not isinstance(fact, dict):
        return_value = {}
        return return_value

    raw_result = fact.get("raw_result")
    if isinstance(raw_result, dict):
        return_value = {
            "global_user_id": str(raw_result.get("global_user_id", "")).strip(),
            "display_name": str(raw_result.get("display_name", "")).strip(),
        }
        return return_value
    return_value = {}
    return return_value


def _is_about_resolved_user_search(task: str) -> bool:
    """Return whether the memory task asks about a resolved user as subject.

    Args:
        task: Slot description produced by the outer-loop supervisor.

    Returns:
        True when the slot is about impressions/opinions/facts of a resolved user,
        not memories sourced by that user.
    """
    task_lower = task.lower()
    references_resolved_user = "user resolved in slot" in task_lower
    asks_about_subject = any(
        marker in task_lower
        for marker in ("about", "impression", "opinion")
    )
    return_value = references_resolved_user and asks_about_subject
    return return_value


def _apply_resolved_subject_user(
    args: dict[str, Any], task: str, context: dict[str, Any]
) -> dict[str, Any]:
    """Adjust memory-search args for third-party subject-user queries.

    ``source_global_user_id`` means the user who sourced/triggered a memory, not
    necessarily the user the memory is about. For "about the user resolved in
    slot N" tasks, keep the resolved user's display name in the semantic query
    and remove the source filter so memories learned from other users remain
    searchable.

    Args:
        args: Tool arguments generated by the LLM.
        task: Slot description produced by the outer-loop supervisor.
        context: Agent context containing known_facts.

    Returns:
        Adjusted tool arguments.
    """
    if not _is_about_resolved_user_search(task):
        return args

    adjusted_args = dict(args)
    adjusted_args.pop("source_global_user_id", None)

    resolved_user = _resolved_user_from_known_facts(task, context)
    display_name = resolved_user.get("display_name", "")
    search_query = str(adjusted_args.get("search_query", "")).strip()
    if display_name and display_name not in search_query:
        adjusted_args["search_query"] = f"{search_query} {display_name}".strip()

    return adjusted_args


async def _generator(task: str, context: dict[str, Any], feedback: str) -> dict[str, Any]:
    """Generate one `search_persistent_memory` argument dict.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Known facts and runtime hints.
        feedback: Judge feedback from the previous attempt, or empty string.

    Returns:
        Normalized arguments for ``search_persistent_memory``.
    """
    system_prompt = SystemMessage(content=_GENERATOR_PROMPT)
    llm_context = project_runtime_context_for_llm(context)
    human_message = HumanMessage(
        content=json.dumps(
            {"task": task, "context": llm_context, "feedback": feedback},
            ensure_ascii=False,
            default=str,
        )
    )
    response = await _generator_llm.ainvoke([system_prompt, human_message])
    result = parse_llm_json_output(response.content)
    if not isinstance(result, dict):
        return_value = {}
        return return_value
    return_value = _normalize_args(result)
    return return_value


async def _tool(args: dict[str, Any]) -> object:
    """Execute `search_persistent_memory` exactly once.

    Args:
        args: Normalized arguments for the tool.

    Returns:
        Tool result or an error dict on invalid arguments.
    """
    try:
        semantic_args = _semantic_tool_args(args)
        semantic_result = await search_persistent_memory.ainvoke(semantic_args)
        semantic_rows = _semantic_rows_from_result(semantic_result)
        keyword_rows = await _keyword_rows_for_anchors(args)
        candidates = merge_hybrid_candidates(
            semantic_rows,
            keyword_rows,
            semantic_only_floor=RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
            selected_limit=RAG_SEARCH_SELECTED_LIMIT,
            source="persistent_memory",
        )
        return_value = _rows_from_candidates(candidates)
        return return_value
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info(f'persistent_memory_search_agent invalid args: {exc}')
        return_value = {"error": f"{type(exc).__name__}: {exc}"}
        return return_value


def _apply_runtime_constraints(
    args: dict[str, Any],
    context: dict[str, Any],
    task: str = "",
) -> dict[str, Any]:
    """Apply runtime-owned source filters and task anchors after generation."""

    constrained = apply_source_memory_runtime_constraints(args, context=context)
    generated_anchors = constrained.get("literal_anchors")
    anchors = generated_anchors if isinstance(generated_anchors, list) else []
    normalized_anchors = _normalize_literal_anchors(anchors)
    for anchor in literal_anchors_from_text(
        task,
        limit=RAG_HYBRID_LITERAL_ANCHOR_LIMIT,
    ):
        if anchor not in normalized_anchors:
            normalized_anchors.append(anchor)
        if len(normalized_anchors) >= RAG_HYBRID_LITERAL_ANCHOR_LIMIT:
            break
    if normalized_anchors:
        constrained["literal_anchors"] = normalized_anchors
    else:
        constrained.pop("literal_anchors", None)

    return constrained


def _semantic_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Remove hybrid-only fields before calling semantic memory retrieval."""

    semantic_args = {
        key: value
        for key, value in args.items()
        if key != "literal_anchors"
    }
    return semantic_args


def _keyword_tool_args(
    args: dict[str, Any],
    anchor: str,
) -> dict[str, Any]:
    """Build keyword memory tool args from shared semantic filters."""

    keyword_args: dict[str, Any] = {
        "keyword": anchor,
        "top_k": args.get("top_k", RAG_SEARCH_DEFAULT_TOP_K),
    }
    source_global_user_id = args.get("source_global_user_id")
    if source_global_user_id:
        keyword_args["source_global_user_id"] = source_global_user_id
    return keyword_args


def _semantic_rows_from_result(result: object) -> list[dict[str, Any]]:
    """Convert semantic memory tool output into hybrid rows."""

    if not isinstance(result, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows = [
        dict(item)
        for item in result
        if isinstance(item, dict)
    ]
    return rows


async def _keyword_rows_for_anchors(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Run keyword memory retrieval for literal anchors generated with the query."""

    anchors = args.get("literal_anchors")
    if not isinstance(anchors, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_text = text_or_empty(anchor)
        if not anchor_text:
            continue
        keyword_result = await search_persistent_memory_keyword.ainvoke(
            _keyword_tool_args(args, anchor_text)
        )
        if not isinstance(keyword_result, list):
            continue
        for item in keyword_result:
            if not isinstance(item, dict):
                continue
            row = dict(item)
            row["method"] = f"keyword:{anchor_text}"
            row["matched_anchors"] = [anchor_text]
            rows.append(row)

    return rows


def _rows_from_candidates(candidates: list[HybridCandidate]) -> list[dict[str, Any]]:
    """Project fused candidates back to ordinary memory rows."""

    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        row = dict(candidate.row)
        row["methods"] = list(candidate.methods)
        row["matched_anchors"] = list(candidate.matched_anchors)
        row["score"] = candidate.score
        row["hybrid_rank"] = rank
        rows.append(row)
    return rows


async def _judge(task: str, result: object) -> tuple[bool, str]:
    """Judge whether the latest semantic memory-search result resolves the slot.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        result: Tool result from the current attempt.

    Returns:
        Tuple of (resolved, feedback).
    """
    system_prompt = SystemMessage(content=_JUDGE_PROMPT)
    llm_result = project_tool_result_for_llm(result)
    human_message = HumanMessage(
        content=json.dumps({"task": task, "result": llm_result}, ensure_ascii=False)
    )
    response = await _judge_llm.ainvoke([system_prompt, human_message])
    verdict = parse_llm_json_output(response.content)
    if not isinstance(verdict, dict):
        return_value = False, "Invalid judge output; make the memory query more concrete."
        return return_value

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return_value = resolved, feedback
    return return_value


class PersistentMemorySearchAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a slot through iterative semantic search over persistent memories.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="persistent_memory_search_agent",
            cache_name=PERSISTENT_MEMORY_SEARCH_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve one slot through iterative semantic search over persistent memories.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Known facts and runtime hints that may constrain the search.
            max_attempts: Maximum generator-tool-judge iterations to attempt.

        Returns:
            Dict with resolved (bool), result (last tool result), and attempts count.
        """
        cache_key = build_persistent_memory_search_cache_key(task, context)
        cached = await self.read_cache(cache_key)
        if cached is not None:
            return_value = self.with_cache_status(
                {"resolved": True, "result": cached, "attempts": 0},
                hit=True,
                reason="hit",
                cache_key=cache_key,
            )
            return return_value

        feedback = ""
        result = None
        resolved = False
        attempt = 0
        args: dict[str, Any] = {}
        cache_stored = False

        for attempt in range(max_attempts):
            args = await _generator(task, context, feedback)
            args = _apply_resolved_subject_user(args, task, context)
            args = _apply_runtime_constraints(args, context, task)
            result = await _tool(args)
            resolved, feedback = await _judge(task, result)
            if resolved:
                break

        if resolved:
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_persistent_memory_search_dependencies(args),
                metadata={},
            )
            cache_stored = True

        if cache_stored:
            cache_reason = "miss_stored"
        elif resolved:
            cache_reason = "miss_not_cacheable"
        else:
            cache_reason = "miss_unresolved"

        return_value = self.with_cache_status(
            {
                "resolved": resolved,
                "result": result,
                "attempts": attempt + 1,
            },
            hit=False,
            reason=cache_reason,
            cache_key=cache_key,
        )
        return return_value


async def _test_main() -> None:
    """Run a manual smoke check for PersistentMemorySearchAgent."""
    agent = PersistentMemorySearchAgent()
    result = await agent.run(
        task="找出和'洗车'有关的持久记忆条目",
        context={"known_facts": []},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task="找出和'洗车'有关的持久记忆条目",
        context={"known_facts": []},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
