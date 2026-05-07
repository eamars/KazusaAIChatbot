"""Validation helpers for bounded RAG query-refinement decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypedDict


MAX_CONTINUATION_DECISIONS_PER_RAG_RUN = 2
MAX_REFINED_QUERY_LENGTH = 1200

_TRUE_STRINGS = {
    "1",
    "true",
    "yes",
    "continue",
}
_DISALLOWED_REFINED_QUERY_PREFIXES = (
    "Live-context:",
    "Conversation-evidence:",
    "Memory-evidence:",
    "Person-context:",
    "Recall:",
    "Web-evidence:",
    "Identity:",
    "User-list:",
    "Relationship:",
    "Profile:",
    "Conversation-aggregate:",
    "Conversation-filter:",
    "Conversation-keyword:",
    "Conversation-semantic:",
    "Memory-search:",
    "Web-search:",
)
_LOW_LEVEL_QUERY_MARKERS = (
    "$match",
    "$search",
    "$vectorsearch",
    "aggregate(",
    "collection:",
    "collection=",
    "db.",
    "find(",
    "mongodb",
    "pipeline:",
    "queryvector",
)
_MISSING_USER_PLACEHOLDER_MARKERS = (
    "ask the user",
    "based on my budget",
    "based on my preference",
    "based on my use case",
    "my budget",
    "my constraints",
    "my preferences",
    "my use case",
    "provide your budget",
    "provide your preferences",
    "user preferences",
    "user's budget",
    '为我推荐',
    '个人偏好',
    '具体用途',
    '我的偏好',
    '我的预算',
    '我的用途',
    '用户偏好',
    '用户尚未提供',
    '用户的偏好',
    '用户的预算',
    '用户的用途',
    '需要用户',
)


class RAGContinuationDecision(TypedDict):
    """Structured decision for bounded refined-query re-entry."""

    should_continue: bool
    refined_query: str
    reason: str


def empty_continuation_decision() -> RAGContinuationDecision:
    """Return the inert decision used when no follow-up retrieval is allowed."""

    decision: RAGContinuationDecision = {
        "should_continue": False,
        "refined_query": "",
        "reason": "",
    }
    return decision


def normalize_continuation_decision(raw: object) -> RAGContinuationDecision:
    """Normalize untrusted refiner output into the public decision shape.

    Args:
        raw: Parsed LLM output or another external value.

    Returns:
        A fully populated continuation decision. Stop decisions cannot carry a
        refined query because only ``should_continue`` may trigger re-entry.
    """

    if not isinstance(raw, Mapping):
        decision = empty_continuation_decision()
        return decision

    should_continue = _bool_value(raw.get("should_continue"))
    refined_query = ""
    if should_continue:
        refined_query = _text(
            raw.get("refined_query"),
            limit=MAX_REFINED_QUERY_LENGTH,
        )

    decision: RAGContinuationDecision = {
        "should_continue": should_continue,
        "refined_query": refined_query,
        "reason": _text(raw.get("reason"), limit=MAX_REFINED_QUERY_LENGTH),
    }
    return decision


def validate_refined_query(
    raw: object,
    *,
    original_query: str,
    previous_refined_queries: Sequence[str],
    continuation_count: int,
    max_continuations: int = MAX_CONTINUATION_DECISIONS_PER_RAG_RUN,
) -> RAGContinuationDecision:
    """Validate whether a normalized decision may re-enter the initializer.

    Args:
        raw: Parsed LLM output or normalized continuation decision.
        original_query: The first user query for this RAG run.
        previous_refined_queries: Refined queries already accepted in this run.
        continuation_count: Prior accepted refined-query decisions.
        max_continuations: Maximum refined-query decisions allowed in this run.

    Returns:
        A decision that may continue only when its refined query is present,
        different from the original and prior refined queries, and shaped like
        natural-language input rather than an executable slot or backend query.
    """

    decision = normalize_continuation_decision(raw)
    if not decision["should_continue"]:
        return decision

    if continuation_count >= max_continuations:
        stopped_decision = _without_refined_query(decision)
        return stopped_decision

    refined_query = decision["refined_query"]
    if not refined_query:
        stopped_decision = _without_refined_query(decision)
        return stopped_decision

    refined_key = _query_key(refined_query)
    original_key = _query_key(original_query)
    previous_keys = _query_key_set(previous_refined_queries)
    if refined_key == original_key or refined_key in previous_keys:
        stopped_decision = _without_refined_query(decision)
        return stopped_decision

    if _looks_like_planned_slot(refined_query):
        stopped_decision = _without_refined_query(decision)
        return stopped_decision

    if _contains_low_level_syntax(refined_query):
        stopped_decision = _without_refined_query(decision)
        return stopped_decision

    if _contains_missing_user_placeholder(refined_query):
        stopped_decision = _without_refined_query(decision)
        return stopped_decision

    return decision


def _without_refined_query(
    decision: RAGContinuationDecision,
) -> RAGContinuationDecision:
    """Preserve trace rationale while preventing initializer re-entry."""

    stopped_decision: RAGContinuationDecision = {
        "should_continue": False,
        "refined_query": "",
        "reason": decision["reason"],
    }
    return stopped_decision


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return_value = value
        return return_value
    if isinstance(value, str):
        return_value = value.strip().casefold() in _TRUE_STRINGS
        return return_value
    return False


def _text(value: object, *, limit: int) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()

    if len(text) > limit:
        text = text[:limit].rstrip()
    return text


def _looks_like_planned_slot(query: str) -> bool:
    looks_like_slot = query.lstrip().startswith(_DISALLOWED_REFINED_QUERY_PREFIXES)
    return looks_like_slot


def _contains_low_level_syntax(query: str) -> bool:
    normalized_query = query.casefold()
    contains_marker = any(
        marker in normalized_query
        for marker in _LOW_LEVEL_QUERY_MARKERS
    )
    return contains_marker


def _contains_missing_user_placeholder(query: str) -> bool:
    normalized_query = query.casefold()
    contains_marker = any(
        marker.casefold() in normalized_query
        for marker in _MISSING_USER_PLACEHOLDER_MARKERS
    )
    return contains_marker


def _query_key(query: str) -> str:
    words = query.split()
    normalized_query = " ".join(words).casefold()
    return normalized_query


def _query_key_set(queries: Sequence[str]) -> set[str]:
    query_keys: set[str] = set()
    for query in queries:
        query_text = _text(query, limit=MAX_REFINED_QUERY_LENGTH)
        if query_text:
            query_keys.add(_query_key(query_text))
    return query_keys
