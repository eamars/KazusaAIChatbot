"""Direct SearXNG JSON search client for web_agent3."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import httpx

from kazusa_ai_chatbot.config import (
    SEARXNG_SEARCH_RESULT_LIMIT,
    SEARXNG_SEARCH_TIMEOUT_SECONDS,
    SEARXNG_URL,
)

_SEARCH_ERROR_CHAR_LIMIT = 800
_SEARCH_FIELD_CHAR_LIMIT = 500
_SEARCH_OUTPUT_CHAR_LIMIT = 12000


def _bounded_text(value: Any, *, limit: int) -> str:
    """Convert external data to prompt-facing text within a character cap."""

    text = str(value or "").strip()
    if len(text) <= limit:
        return text

    clipped_text = text[:limit].rstrip()
    return_value = f"{clipped_text}..."
    return return_value


def _error_message(prefix: str, exc: BaseException) -> str:
    """Build a bounded error observation that includes concrete exception text."""

    message = f"Error: {prefix}: {exc}"
    if len(message) <= _SEARCH_ERROR_CHAR_LIMIT:
        return message

    clipped_message = message[:_SEARCH_ERROR_CHAR_LIMIT].rstrip()
    return_value = f"{clipped_message}..."
    return return_value


def _format_result_row(row: Mapping[str, Any]) -> str:
    """Format one SearXNG result row as a compact text record."""

    title = _bounded_text(row.get("title", ""), limit=_SEARCH_FIELD_CHAR_LIMIT)
    url = _bounded_text(row.get("url", ""), limit=_SEARCH_FIELD_CHAR_LIMIT)
    content_value = row.get("content")
    snippet_value = row.get("snippet")
    if content_value:
        snippet_source = content_value
    else:
        snippet_source = snippet_value
    snippet = _bounded_text(snippet_source, limit=_SEARCH_FIELD_CHAR_LIMIT)

    lines = [
        f"Title: {title}",
        f"URL: {url}",
        f"Snippet: {snippet}",
    ]
    engine = _bounded_text(row.get("engine", ""), limit=_SEARCH_FIELD_CHAR_LIMIT)
    if engine:
        lines.append(f"Engine: {engine}")

    score = row.get("score")
    if score is not None and score != "":
        score_text = _bounded_text(score, limit=_SEARCH_FIELD_CHAR_LIMIT)
        lines.append(f"Score: {score_text}")

    return_value = "\n".join(lines)
    return return_value


def _format_search_results(payload: Mapping[str, Any]) -> str:
    """Format SearXNG JSON results into bounded text for the web agent."""

    raw_results = payload.get("results", [])
    if not isinstance(raw_results, list):
        return_value = "Error: invalid SearXNG JSON: results is not a list"
        return return_value

    rows: list[str] = []
    for raw_row in raw_results[:SEARXNG_SEARCH_RESULT_LIMIT]:
        if not isinstance(raw_row, Mapping):
            continue
        row_text = _format_result_row(raw_row)
        rows.append(row_text)

    if not rows:
        return_value = "No results found."
        return return_value

    formatted_text = "\n\n".join(rows)
    if len(formatted_text) <= _SEARCH_OUTPUT_CHAR_LIMIT:
        return formatted_text

    clipped_text = formatted_text[:_SEARCH_OUTPUT_CHAR_LIMIT].rstrip()
    return_value = f"{clipped_text}..."
    return return_value


async def web_search(
    query: str,
    pageno: int = 1,
    time_range: str = "",
    language: str = "",
) -> str:
    """Search the configured SearXNG instance through its JSON API.

    Args:
        query: Search query text.
        pageno: SearXNG page number.
        time_range: Optional SearXNG time range parameter.
        language: Optional SearXNG language parameter.

    Returns:
        Bounded prompt-facing search results, or a bounded error observation.
    """
    if not SEARXNG_URL:
        return_value = (
            "Error: SearXNG search unavailable: SEARXNG_URL is not configured."
        )
        return return_value

    params: dict[str, str | int] = {
        "q": query,
        "format": "json",
        "pageno": pageno,
        "safesearch": 0,
    }
    time_range_value = time_range.strip()
    if time_range_value:
        params["time_range"] = time_range_value
    language_value = language.strip()
    if language_value:
        params["language"] = language_value
    search_url = f"{SEARXNG_URL}/search"

    try:
        async with httpx.AsyncClient(
            timeout=SEARXNG_SEARCH_TIMEOUT_SECONDS,
        ) as client:
            response = await client.get(search_url, params=params)
            response.raise_for_status()
    except httpx.TimeoutException as exc:
        return_value = _error_message("SearXNG search timed out", exc)
        return return_value
    except httpx.HTTPStatusError as exc:
        return_value = _error_message("SearXNG search HTTP error", exc)
        return return_value
    except httpx.RequestError as exc:
        return_value = _error_message("SearXNG search request failed", exc)
        return return_value

    try:
        payload = response.json()
    except ValueError as exc:
        return_value = _error_message("SearXNG search returned invalid JSON", exc)
        return return_value

    if not isinstance(payload, Mapping):
        return_value = "Error: invalid SearXNG JSON: top-level value is not an object"
        return return_value

    formatted_results = _format_search_results(payload)
    return formatted_results
