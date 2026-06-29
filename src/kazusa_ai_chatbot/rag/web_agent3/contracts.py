"""Shared contracts for the routed web_agent3 helper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

_DEFAULT_EXPECTED_RESPONSE = "返回能直接解决当前槽位的来源扎根网页证据。"
_ROUTER_ACTIONS = ("search", "read", "stop")
_ROUTER_SOURCES = ("web_read",)
_ROUTER_SOURCE_ACTIONS = {
    "web_read": ("read",),
}
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


def _source_supports_action(
    source: str,
    action: str,
    source_actions: dict[str, tuple[str, ...]],
) -> bool:
    """Return whether an enabled source executes the requested action."""
    supported_actions = source_actions.get(source, ())
    supports_action = action in supported_actions
    return supports_action


def _is_http_url(raw_value: str) -> bool:
    """Return whether a router query is a direct HTTP(S) URL target."""
    lowered_value = raw_value.lower()
    is_http_url = (
        lowered_value.startswith("http://")
        or lowered_value.startswith("https://")
    )
    return is_http_url


def _stop_router_decision() -> _RouterDecision:
    """Build the graph-local stop decision placeholder."""
    decision = _RouterDecision(
        action="stop",
        source="web_read",
        query="",
    )
    return decision


def _normalize_router_decision(
    raw_decision: dict[str, Any],
    *,
    fallback_query: str,
    valid_sources: tuple[str, ...] | None = None,
    source_actions: dict[str, tuple[str, ...]] | None = None,
) -> _RouterDecision:
    """Validate the router LLM decision and discard unsupported fields.

    Args:
        raw_decision: Parsed LLM JSON object.
        fallback_query: Task text used when a searchable decision omits query.

    Returns:
        Router decision containing only action, source, and query.
    """
    source_names = _ROUTER_SOURCES if valid_sources is None else valid_sources
    enabled_sources = set(source_names)
    action_map = _ROUTER_SOURCE_ACTIONS if source_actions is None else source_actions
    raw_action = _text_field(raw_decision.get("action")).lower()
    raw_source = _text_field(raw_decision.get("source")).lower()
    query = _text_field(raw_decision.get("query"))

    if raw_action in _ROUTER_ACTIONS:
        action = raw_action
    else:
        action = "search"

    if action == "stop":
        decision = _stop_router_decision()
        return decision

    if not query:
        if (
            "web_search" in enabled_sources
            and _source_supports_action("web_search", "search", action_map)
        ):
            source = "web_search"
            action = "search"
            query = fallback_query.strip()
        else:
            decision = _stop_router_decision()
            return decision

    elif action == "read":
        if (
            raw_source in enabled_sources
            and _source_supports_action(raw_source, "read", action_map)
        ):
            source = raw_source
        elif (
            "web_read" in enabled_sources
            and _source_supports_action("web_read", "read", action_map)
            and _is_http_url(query)
        ):
            source = "web_read"
        else:
            decision = _stop_router_decision()
            return decision

    elif action == "search":
        if raw_source == "nhentai":
            if (
                raw_source in enabled_sources
                and _source_supports_action(raw_source, "search", action_map)
            ):
                source = raw_source
            else:
                decision = _stop_router_decision()
                return decision
        elif (
            raw_source in enabled_sources
            and _source_supports_action(raw_source, "search", action_map)
        ):
            source = raw_source
        elif (
            "web_search" in enabled_sources
            and _source_supports_action("web_search", "search", action_map)
        ):
            source = "web_search"
        else:
            decision = _stop_router_decision()
            return decision

    if not query:
        if source == "web_search":
            query = fallback_query.strip()
        if not query:
            decision = _stop_router_decision()
            return decision

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
