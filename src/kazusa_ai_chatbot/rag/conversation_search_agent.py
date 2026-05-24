"""RAG helper agent: semantic conversation search."""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from kazusa_ai_chatbot.config import (
    CONVERSATION_SEARCH_DEFAULT_TOP_K,
    CONVERSATION_SEARCH_MAX_TOP_K,
    RAG_HYBRID_LITERAL_ANCHOR_LIMIT,
    RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT,
    RAG_HYBRID_NEIGHBOR_SEED_LIMIT,
    RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES,
    RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
    RAG_SEARCH_SELECTED_LIMIT,
    RAG_SUBAGENT_LLM_API_KEY,
    RAG_SUBAGENT_LLM_BASE_URL,
    RAG_SUBAGENT_LLM_MODEL,
)
from kazusa_ai_chatbot.db import get_conversation_history
from kazusa_ai_chatbot.rag.memory_retrieval_tools import (
    conversation_message_payload,
    search_conversation,
    search_conversation_keyword,
)
from kazusa_ai_chatbot.rag.cache2_policy import (
    CONVERSATION_SEARCH_CACHE_NAME,
    build_conversation_search_cache_key,
    build_conversation_search_dependencies,
    is_closed_historical_range,
)
from kazusa_ai_chatbot.rag.helper_agent import BaseRAGHelperAgent
from kazusa_ai_chatbot.rag.hybrid_retrieval import (
    HybridCandidate,
    merge_hybrid_candidates,
    select_neighbor_seed_candidates,
)
from kazusa_ai_chatbot.rag.prompt_projection import (
    project_runtime_context_for_llm,
    project_tool_result_for_llm,
)
from kazusa_ai_chatbot.rag.search_runtime import (
    apply_conversation_runtime_constraints,
)
from kazusa_ai_chatbot.time_boundary import (
    local_llm_datetime_to_storage_utc_iso,
    parse_storage_utc_datetime,
)
from kazusa_ai_chatbot.utils import get_llm, parse_llm_json_output, text_or_empty

logger = logging.getLogger(__name__)

_RAW_CONVERSATION_STORAGE_KEYS = (
    "_id",
    "embedding",
    "raw_wire_text",
    "base64_data",
)

_GENERATOR_PROMPT = (
    """\
You generate retrieval parameters only for `search_conversation`.

# Scope
- Produce one high-quality semantic search argument set for the current slot.
- You may generate parameters only for `search_conversation`.
- `search_query` must be a natural-language semantic query, not a keyword list.
- `literal_anchors` is optional. Include only proper nouns, technical terms,
  short phrases, URLs, filenames, or quoted text that must match literally.
  Do not split a full sentence into a word list.
- If context already provides platform, platform_channel_id, global_user_id,
  or time bounds, use those trusted constraints.
- `context.original_query` is the decontextualized user request. When `task`
  only locates an anchor, screenshot, link, or timestamp, still include the
  surrounding facts, follow-up discussion, and speaker relationships requested
  by `original_query` in `search_query`.
- `feedback` comes from the previous judge pass. If it says too broad, wrong
  angle, or irrelevant result, rewrite the query angle instead of repeating it.
- Use English for generated control/status text. Preserve literal anchors,
  names, quotes, URLs, filenames, and source message text in their original
  source language.

# Generation Procedure
1. Read `task` to identify the semantic conversation evidence needed.
2. Read `context.original_query`, `context.current_slot`, and `context.known_facts`
   for broader information needs that the narrowed slot may omit.
3. Read `context` for platform, channel, user, time, and known-fact constraints.
4. Read `feedback`; if present, change the search angle instead of repeating the previous query.
5. Write one natural-language `search_query`, then add only filters supported by context.

# Input Format
{{
  "task": "slot description from the outer RAG supervisor",
  "context": "known facts and runtime hints",
  "feedback": "previous judge feedback, or empty string"
}}

# Output Format
Return valid JSON only:
{{
  "search_query": "string",
  "literal_anchors": ["string, optional; at most 5 anchors"],
  "global_user_id": "string or omitted",
  "top_k": {default_top_k},
  "platform": "string or omitted",
  "platform_channel_id": "string or omitted",
  "from_timestamp": "local YYYY-MM-DD HH:MM or omitted",
  "to_timestamp": "local YYYY-MM-DD HH:MM or omitted"
}}
"""
).format(default_top_k=CONVERSATION_SEARCH_DEFAULT_TOP_K)
_generator_llm = get_llm(
    temperature=0.0,
    top_p=1.0,
    model=RAG_SUBAGENT_LLM_MODEL,
    base_url=RAG_SUBAGENT_LLM_BASE_URL,
    api_key=RAG_SUBAGENT_LLM_API_KEY,
)

_JUDGE_PROMPT = """\
You judge whether a `search_conversation` result resolves the current slot.

# Task
- Decide whether the current result is enough to resolve the slot.
- If unresolved, `feedback` must give an executable correction for the next
  search query or filters.

# Generation Procedure
1. Read `task` and identify what evidence would resolve it.
2. Read `context.original_query` and `context.current_slot`; if they ask for
   surrounding conversation, follow-up discussion, or additional entities that
   the narrowed `task` omits, require those facts too.
3. Inspect `result` for directly relevant messages, source metadata, and errors.
4. Return resolved=true only when the result answers the slot and does not
   drop essential broader context visible in `context`.
5. If unresolved, write concrete feedback for the next search query or filters.

# Input Format
{
  "task": "slot description from the outer RAG supervisor",
  "context": "projected runtime context, including original_query/current_slot when available",
  "result": "tool result from search_conversation"
}

# Common Feedback Directions
- Query too broad; use a more specific semantic description.
- Query angle wrong; focus on person, link, event, or requested detail.
- Missing filters; use known user or time range.
- Returned messages irrelevant; rewrite as "opinion about X" or "mentions X".
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


def _normalize_top_k(value: object) -> int:
    """Normalize semantic conversation-search top-k to configured bounds."""

    if not isinstance(value, int) or isinstance(value, bool):
        return_value = CONVERSATION_SEARCH_DEFAULT_TOP_K
        return return_value

    if value < CONVERSATION_SEARCH_DEFAULT_TOP_K:
        return_value = CONVERSATION_SEARCH_DEFAULT_TOP_K
        return return_value

    return_value = min(value, CONVERSATION_SEARCH_MAX_TOP_K)
    return return_value


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


def _normalize_args(raw_args: dict[str, Any]) -> dict[str, Any]:
    """Keep only valid `search_conversation` arguments with safe scalar coercion.

    Args:
        raw_args: Raw dict from the generator LLM.

    Returns:
        Validated argument dict for ``search_conversation``.
    """
    args: dict[str, Any] = {}

    search_query = text_or_empty(raw_args.get("search_query"))
    if search_query:
        args["search_query"] = search_query

    literal_anchors = _normalize_literal_anchors(raw_args.get("literal_anchors"))
    if literal_anchors:
        args["literal_anchors"] = literal_anchors

    args["top_k"] = _normalize_top_k(raw_args.get("top_k"))

    for key in (
        "global_user_id",
        "platform",
        "platform_channel_id",
    ):
        raw_val = raw_args.get(key)
        if raw_val is None:
            continue
        value = text_or_empty(raw_val)
        if value:
            args[key] = value

    for key in ("from_timestamp", "to_timestamp"):
        value = text_or_empty(raw_args.get(key))
        if not value:
            continue
        try:
            args[key] = local_llm_datetime_to_storage_utc_iso(value)
        except ValueError as exc:
            logger.debug(f"Dropping invalid {key} from LLM output: {exc}")

    return args


async def _generator(task: str, context: dict[str, Any], feedback: str) -> dict[str, Any]:
    """Generate one `search_conversation` argument dict for the current attempt.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        context: Known facts and runtime hints.
        feedback: Judge feedback from the previous attempt, or empty string.

    Returns:
        Normalized arguments for ``search_conversation``.
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
    """Execute `search_conversation` exactly once and return the result.

    Args:
        args: Normalized arguments for the tool.

    Returns:
        Tool result or an error dict on invalid arguments.
    """
    try:
        return_value = await run_hybrid_conversation_search(args)
        return return_value
    except (TypeError, ValueError, ValidationError) as exc:
        logger.info(f'conversation_search_agent invalid args: {exc}')
        return_value = {"error": f"{type(exc).__name__}: {exc}"}
        return return_value


async def run_hybrid_conversation_search(
    args: dict[str, Any],
    *,
    semantic_only_floor: float = RAG_HYBRID_SEMANTIC_ONLY_SCORE_FLOOR,
    selected_limit: int = RAG_SEARCH_SELECTED_LIMIT,
    neighbor_seed_limit: int = RAG_HYBRID_NEIGHBOR_SEED_LIMIT,
    neighbor_message_limit: int = RAG_HYBRID_NEIGHBOR_MESSAGE_LIMIT,
    neighbor_window_minutes: int = RAG_HYBRID_NEIGHBOR_WINDOW_MINUTES,
) -> list[dict[str, Any]]:
    """Run the production hybrid conversation retrieval pipeline.

    Args:
        args: Normalized search arguments, including semantic query, optional
            literal anchors, trusted filters, and top-k.
        semantic_only_floor: Minimum score for semantic-only rows.
        selected_limit: Maximum fused rows to return.
        neighbor_seed_limit: Number of direct rows allowed to seed neighbors.
        neighbor_message_limit: Number of neighboring rows to fetch per side.
        neighbor_window_minutes: Time window around each direct row.

    Returns:
        Ranked row dictionaries with hybrid provenance fields.
    """

    semantic_args = _semantic_tool_args(args)
    semantic_result = await search_conversation.ainvoke(semantic_args)
    semantic_rows = _semantic_rows_from_result(semantic_result)
    keyword_rows = await _keyword_rows_for_anchors(args)
    candidates = merge_hybrid_candidates(
        semantic_rows,
        keyword_rows,
        semantic_only_floor=semantic_only_floor,
        selected_limit=selected_limit,
        source="conversation",
    )
    neighbor_rows = await _conversation_neighbor_rows(
        candidates,
        args,
        semantic_only_floor=semantic_only_floor,
        seed_limit=neighbor_seed_limit,
        message_limit=neighbor_message_limit,
        window_minutes=neighbor_window_minutes,
    )
    if neighbor_rows:
        candidates = merge_hybrid_candidates(
            semantic_rows,
            keyword_rows,
            neighbor_rows,
            semantic_only_floor=semantic_only_floor,
            selected_limit=selected_limit,
            source="conversation",
        )
    rows = _rows_from_candidates(candidates)
    return rows


def _semantic_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Remove hybrid-only fields before calling the semantic tool."""

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
    """Build keyword tool args from shared semantic filters."""

    keyword_args: dict[str, Any] = {
        "keyword": anchor,
        "top_k": args.get("top_k", CONVERSATION_SEARCH_DEFAULT_TOP_K),
    }
    for key in (
        "global_user_id",
        "platform",
        "platform_channel_id",
        "from_timestamp",
        "to_timestamp",
    ):
        value = args.get(key)
        if value:
            keyword_args[key] = value
    return keyword_args


def _semantic_rows_from_result(result: object) -> list[dict[str, Any]]:
    """Convert semantic tool output into hybrid rows."""

    rows: list[dict[str, Any]] = []
    if not isinstance(result, list):
        return rows

    for item in result:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            score, message = item
            if not isinstance(message, dict):
                continue
            row = _conversation_result_row(message)
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                row["score"] = float(score)
            rows.append(row)
            continue
        if isinstance(item, dict):
            row = _conversation_result_row(item)
            rows.append(row)

    return rows


async def _keyword_rows_for_anchors(args: dict[str, Any]) -> list[dict[str, Any]]:
    """Run keyword retrieval for literal anchors generated with the query."""

    anchors = args.get("literal_anchors")
    if not isinstance(anchors, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows: list[dict[str, Any]] = []
    for anchor in anchors:
        anchor_text = text_or_empty(anchor)
        if not anchor_text:
            continue
        keyword_result = await search_conversation_keyword.ainvoke(
            _keyword_tool_args(args, anchor_text)
        )
        if not isinstance(keyword_result, list):
            continue
        for item in keyword_result:
            if not isinstance(item, dict):
                continue
            row = _conversation_result_row(item)
            row["method"] = f"keyword:{anchor_text}"
            row["matched_anchors"] = [anchor_text]
            rows.append(row)

    return rows


async def _conversation_neighbor_rows(
    candidates: list[HybridCandidate],
    args: dict[str, Any],
    *,
    semantic_only_floor: float,
    seed_limit: int,
    message_limit: int,
    window_minutes: int,
) -> list[dict[str, Any]]:
    """Fetch narrow neighboring context around selected direct evidence."""

    anchors_present = bool(args.get("literal_anchors"))
    seeds = select_neighbor_seed_candidates(
        candidates,
        keyword_rows_present=anchors_present,
        semantic_only_floor=semantic_only_floor,
        seed_limit=seed_limit,
    )
    rows_by_identity: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        from_timestamp, to_timestamp = _neighbor_time_bounds(
            seed.row,
            window_minutes=window_minutes,
        )
        if not from_timestamp or not to_timestamp:
            continue
        seed_timestamp = text_or_empty(seed.row.get("timestamp"))
        if not seed_timestamp:
            continue
        side_limit = max(1, message_limit)
        platform = text_or_empty(seed.row.get("platform"))
        if not platform:
            platform = text_or_empty(args.get("platform"))
        platform_channel_id = text_or_empty(seed.row.get("platform_channel_id"))
        if not platform_channel_id:
            platform_channel_id = text_or_empty(args.get("platform_channel_id"))
        before_docs = await get_conversation_history(
            platform=platform or None,
            platform_channel_id=platform_channel_id or None,
            limit=side_limit,
            from_timestamp=from_timestamp,
            to_timestamp=seed_timestamp,
            sort_direction=-1,
        )
        after_docs = await get_conversation_history(
            platform=platform or None,
            platform_channel_id=platform_channel_id or None,
            limit=side_limit,
            from_timestamp=seed_timestamp,
            to_timestamp=to_timestamp,
            sort_direction=1,
        )
        for doc in before_docs + after_docs:
            if not isinstance(doc, dict):
                continue
            candidate = _conversation_result_row(doc)
            identity = text_or_empty(candidate.get("platform_message_id"))
            if not identity:
                identity = text_or_empty(candidate.get("conversation_row_id"))
            if not identity:
                identity = text_or_empty(candidate.get("_id"))
            if not identity:
                identity = f"{candidate.get('timestamp', '')}:{len(rows_by_identity)}"
            rows_by_identity[identity] = candidate

    rows = sorted(
        rows_by_identity.values(),
        key=lambda row: text_or_empty(row.get("timestamp")),
    )
    return rows


def _neighbor_time_bounds(
    row: dict[str, Any],
    *,
    window_minutes: int,
) -> tuple[str, str]:
    """Return UTC ISO bounds around one candidate timestamp."""

    timestamp = text_or_empty(row.get("timestamp"))
    if not timestamp:
        return_value = "", ""
        return return_value

    try:
        center = parse_storage_utc_datetime(timestamp)
    except ValueError:
        return_value = "", ""
        return return_value
    window = timedelta(minutes=window_minutes)
    from_timestamp = (center - window).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    to_timestamp = (center + window).strftime("%Y-%m-%dT%H:%M:%S+00:00")
    return_value = from_timestamp, to_timestamp
    return return_value


def _conversation_result_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project one conversation row into the search-agent result contract."""

    if "body_text" not in row:
        return _strip_raw_conversation_storage_fields(row)

    payload = conversation_message_payload(row)
    for key in (
        "error",
        "method",
        "methods",
        "matched_anchors",
        "score",
        "hybrid_rank",
    ):
        if key in row:
            payload[key] = row[key]

    return payload


def _strip_raw_conversation_storage_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Drop raw storage fields from non-standard conversation rows."""

    projected = dict(row)
    if "_id" in projected and "conversation_row_id" not in projected:
        projected["conversation_row_id"] = str(projected["_id"])
    for key in _RAW_CONVERSATION_STORAGE_KEYS:
        projected.pop(key, None)
    return projected


def _rows_from_candidates(candidates: list[HybridCandidate]) -> list[dict[str, Any]]:
    """Project fused candidates back to ordinary row dictionaries."""

    rows: list[dict[str, Any]] = []
    for rank, candidate in enumerate(candidates, start=1):
        row = _conversation_result_row(candidate.row)
        row["methods"] = list(candidate.methods)
        row["matched_anchors"] = list(candidate.matched_anchors)
        row["score"] = candidate.score
        row["hybrid_rank"] = rank
        rows.append(row)
    return rows


def _apply_context_top_k(args: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
    """Apply a bounded internal top-k override from trusted worker context."""

    override = context.get("conversation_search_top_k")
    if override is None:
        return dict(args)

    scoped_args = dict(args)
    scoped_args["top_k"] = _normalize_top_k(override)
    return scoped_args


def _apply_runtime_constraints(
    args: dict[str, Any],
    task: str,
    context: dict[str, Any],
) -> dict[str, Any]:
    """Apply runtime-owned filters and task anchors after LLM generation."""

    constrained = apply_conversation_runtime_constraints(
        args,
        context=context,
        task=task,
        literal_anchor_limit=RAG_HYBRID_LITERAL_ANCHOR_LIMIT,
    )
    return constrained


async def _judge(
    task: str,
    result: object,
    context: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Judge whether the latest semantic search result resolves the slot.

    Args:
        task: Slot description produced by the outer-loop dispatcher.
        result: Tool result from the current attempt.
        context: Runtime context carrying the broader original query.

    Returns:
        Tuple of (resolved, feedback).
    """
    system_prompt = SystemMessage(content=_JUDGE_PROMPT)
    llm_result = project_tool_result_for_llm(result)
    llm_context = project_runtime_context_for_llm(context or {})
    human_message = HumanMessage(
        content=json.dumps(
            {
                "task": task,
                "context": llm_context,
                "result": llm_result,
            },
            ensure_ascii=False,
        )
    )
    response = await _judge_llm.ainvoke([system_prompt, human_message])
    verdict = parse_llm_json_output(response.content)
    if not isinstance(verdict, dict):
        return_value = False, "Invalid judge output; make the semantic query more specific."
        return return_value

    resolved = bool(verdict.get("resolved", False))
    feedback = str(verdict.get("feedback", "")).strip()
    return_value = resolved, feedback
    return return_value


class ConversationSearchAgent(BaseRAGHelperAgent):
    """RAG helper agent that resolves a slot through iterative semantic conversation search.

    Args:
        cache_runtime: Optional cache runtime override for tests or local tools.
    """

    def __init__(self, *, cache_runtime=None) -> None:
        super().__init__(
            name="conversation_search_agent",
            cache_name=CONVERSATION_SEARCH_CACHE_NAME,
            cache_runtime=cache_runtime,
        )

    async def run(
        self,
        task: str,
        context: dict[str, Any],
        max_attempts: int = 3,
    ) -> dict[str, Any]:
        """Resolve one slot through iterative semantic search over conversation history.

        Args:
            task: Slot description produced by the outer-loop supervisor.
            context: Known facts and runtime hints that may constrain the search.
            max_attempts: Maximum generator-tool-judge iterations to attempt.

        Returns:
            Dict with resolved (bool), result (last tool result), and attempts count.
        """
        cache_key = build_conversation_search_cache_key(task, context)
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
            args = _apply_runtime_constraints(args, task, context)
            args = _apply_context_top_k(args, context)
            result = await _tool(args)
            resolved, feedback = await _judge(task, result, context)
            if resolved:
                break

        if resolved and is_closed_historical_range(args):
            await self.write_cache(
                cache_key=cache_key,
                result=result,
                dependencies=build_conversation_search_dependencies(args, context),
                metadata={},
            )
            cache_stored = True

        if cache_stored:
            cache_reason = "miss_stored"
        elif resolved:
            cache_reason = "miss_open_conversation_range"
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
    """Run a manual smoke check for ConversationSearchAgent."""
    agent = ConversationSearchAgent()
    result = await agent.run(
        task="最近提到的小红书链接",
        context={
            "platform": "qq",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))

    result = await agent.run(
        task="关于vibe coding的话题",
        context={
            "platform": "qq",
            "known_facts": [],
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import asyncio

    asyncio.run(_test_main())
