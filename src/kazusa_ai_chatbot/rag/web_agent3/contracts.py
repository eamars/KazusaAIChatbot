"""Shared contracts for the routed web_agent3 helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

_DEFAULT_EXPECTED_RESPONSE = "返回能直接解决当前槽位的来源扎根网页证据。"
_DUMMY_PROVIDER_FIXME = (
    "FIXME(web_agent3): replace no-search-data placeholder with provider API client "
    "in a future approved plan."
)
_ROUTER_ACTIONS = ("search", "read", "stop")
_ROUTER_SOURCES = ("generic", "bilibili", "youtube", "nhentai")
_VALID_STATUS_ORDER = {
    "not_found": 0,
    "partial": 1,
    "success": 2,
}

_RouterAction = Literal["search", "read", "stop"]


@dataclass(frozen=True)
class _RouterDecision:
    """Validated source/action decision produced by the router LLM."""

    action: _RouterAction
    source: str
    query: str


@dataclass(frozen=True)
class _WebSearchItem:
    """One bounded web search result item used by comparison fixtures."""

    title: str
    url: str
    snippet: str
    source: str


@dataclass(frozen=True)
class _WebToolResult:
    """Normalized evidence fixture used by comparison and focused tests."""

    resolved: bool
    operation: str
    query: str | None
    url: str | None
    title: str | None
    description: str | None
    content: str
    items: list[_WebSearchItem]
    delegation_reason: str | None
    missing_context: list[str]
    error: str | None
    truncated: bool = False
    headings: list[str] = field(default_factory=list)


def _text_field(value: object) -> str:
    """Return a stripped string for external LLM fields."""
    if not isinstance(value, str):
        return_value = ""
        return return_value
    return_value = value.strip()
    return return_value


def _normalize_router_decision(
    raw_decision: dict[str, Any],
    *,
    fallback_query: str,
    valid_sources: tuple[str, ...] | None = None,
) -> _RouterDecision:
    """Validate the router LLM decision and discard unsupported fields.

    Args:
        raw_decision: Parsed LLM JSON object.
        fallback_query: Task text used when a searchable decision omits query.

    Returns:
        Router decision containing only action, source, and query.
    """
    source_names = valid_sources or _ROUTER_SOURCES
    raw_action = _text_field(raw_decision.get("action")).lower()
    raw_source = _text_field(raw_decision.get("source")).lower()
    query = _text_field(raw_decision.get("query"))

    if raw_action in _ROUTER_ACTIONS:
        action = raw_action
    else:
        action = "search"

    if raw_source in source_names:
        source = raw_source
    else:
        source = "generic"

    if action == "stop":
        query = ""
    elif not query:
        source = "generic"
        action = "search"
        query = fallback_query.strip()

    decision = _RouterDecision(action=action, source=source, query=query)
    return decision


def _router_decision_to_dict(decision: _RouterDecision) -> dict[str, str]:
    """Serialize a router decision to its minimal state shape."""
    payload = {
        "action": decision.action,
        "source": decision.source,
        "query": decision.query,
    }
    return payload


def _router_decision_from_state(raw_decision: dict[str, str]) -> _RouterDecision:
    """Rebuild a validated router decision from graph state."""
    decision = _RouterDecision(
        action=cast(_RouterAction, raw_decision["action"]),
        source=raw_decision["source"],
        query=raw_decision["query"],
    )
    return decision


def _limit_status(
    status: Literal["success", "partial", "not_found"],
    max_status: Literal["success", "partial", "not_found"],
) -> Literal["success", "partial", "not_found"]:
    """Clamp a status to the maximum evidence strength allowed by metadata."""
    status_rank = _VALID_STATUS_ORDER[status]
    max_rank = _VALID_STATUS_ORDER[max_status]
    if status_rank <= max_rank:
        return_value = status
        return return_value

    return_value = max_status
    return return_value
