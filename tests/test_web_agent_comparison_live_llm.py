"""Apple-to-apple real LLM comparison for web_search_agent2 and web_agent3."""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.config import WEB_SEARCH_LLM_BASE_URL, WEB_SEARCH_LLM_MODEL
from kazusa_ai_chatbot.rag import web_agent3 as web_agent3_module
from kazusa_ai_chatbot.rag import web_search_agent as web_agent2_module
from kazusa_ai_chatbot.rag.web_agent3 import searxng_tools as web_agent3_tools
from tests.llm_trace import write_llm_trace

pytestmark = [pytest.mark.asyncio, pytest.mark.live_llm]

logger = logging.getLogger(__name__)

_TRACE_SUITE = "web_agent2_vs_web_agent3_public_run_live_llm"
_REFERENCE_CONTEXT = {
    "platform": "debug",
    "local_time_context": {
        "current_local_datetime": "2026-05-27 12:00",
        "current_local_weekday": "Wednesday",
    },
}

_COMPARISON_CASES: dict[str, dict[str, Any]] = {
    "official_python_release": {
        "task": (
            "Web-evidence: summarize the latest Python release status from "
            "official evidence. Prefer https://www.python.org/downloads/."
        ),
        "operation": "fetch_url",
        "url": "https://www.python.org/downloads/",
        "title": "Download Python",
        "description": "Official Python downloads page.",
        "content": (
            "Python.org downloads page lists Python 3.13.3 as a production "
            "release and links to release notes at https://www.python.org/downloads/."
        ),
        "quality_target": "Preserve version, source URL, and official-source caveat.",
    },
    "search_results_only": {
        "task": "Web-evidence: find source leads about SearXNG JSON API behavior.",
        "operation": "search",
        "query": "SearXNG JSON API format results field",
        "items": [
            {
                "title": "SearXNG search API documentation",
                "url": "https://docs.searxng.org/dev/search_api.html",
                "snippet": "The search API accepts format=json and returns a results list.",
                "source": "docs",
            },
            {
                "title": "SearXNG settings",
                "url": "https://docs.searxng.org/admin/settings.html",
                "snippet": "Result engines and categories can be configured by operators.",
                "source": "docs",
            },
        ],
        "quality_target": "State that evidence is snippet-level and preserve both URLs.",
    },
    "youtube_description": {
        "task": (
            "Web-evidence: summarize the YouTube page description for "
            "https://www.youtube.com/watch?v=example."
        ),
        "operation": "fetch_url",
        "url": "https://www.youtube.com/watch?v=example",
        "title": "Example Tech Talk",
        "description": "Video page for a conference talk.",
        "content": (
            "The page title is Example Tech Talk. The description says the "
            "speaker explains local-first agent architecture and tool routing. "
            "Source URL: https://www.youtube.com/watch?v=example"
        ),
        "quality_target": "Avoid claiming transcript-level detail; identify it as page description.",
    },
    "bilibili_description": {
        "task": (
            "Web-evidence: summarize the Bilibili video page description for "
            "https://www.bilibili.com/video/BV1example."
        ),
        "operation": "fetch_url",
        "url": "https://www.bilibili.com/video/BV1example",
        "title": "Local tool router demo",
        "description": "Bilibili video page for a local tool routing demo.",
        "content": (
            "The Bilibili page describes a software demo showing a local tool "
            "router dispatching a web task to site-specific handlers. "
            "Source URL: https://www.bilibili.com/video/BV1example"
        ),
        "quality_target": "Preserve Bilibili URL and partial-page limitation.",
    },
    "nhentai_api_docs": {
        "task": (
            "Web-evidence: summarize what https://nhentai.net/api/v2/docs "
            "exposes without using gallery content."
        ),
        "operation": "fetch_url",
        "url": "https://nhentai.net/api/v2/docs",
        "title": "API Docs",
        "description": "API documentation page.",
        "content": (
            "The API docs page describes endpoints for gallery metadata, search "
            "results, tags, pages, and image dimensions. It is documentation "
            "only, not gallery content. Source URL: https://nhentai.net/api/v2/docs"
        ),
        "quality_target": "Keep the answer about API docs and avoid adult-content details.",
    },
    "redirected_docs_page": {
        "task": (
            "Web-evidence: summarize the documentation page at "
            "https://example.dev/docs/latest after redirect."
        ),
        "operation": "fetch_url",
        "url": "https://example.dev/docs/latest",
        "title": "Current Documentation",
        "description": "Redirect target for current docs.",
        "content": (
            "The current documentation page says version 2.4 is the maintained "
            "release. The original docs/latest URL redirects to "
            "https://example.dev/docs/2.4/."
        ),
        "quality_target": "Mention redirect target and maintained version.",
    },
    "conflicting_snippets": {
        "task": "Web-evidence: compare current product price claims from search results.",
        "operation": "search",
        "query": "ExamplePhone current price",
        "items": [
            {
                "title": "Retailer A ExamplePhone listing",
                "url": "https://retailer-a.example/phone",
                "snippet": "ExamplePhone listed at USD 799, updated May 2026.",
                "source": "retailer-a",
            },
            {
                "title": "Retailer B ExamplePhone listing",
                "url": "https://retailer-b.example/phone",
                "snippet": "ExamplePhone listed at USD 849, promotion ends soon.",
                "source": "retailer-b",
            },
        ],
        "quality_target": "Do not collapse conflicting prices into one value.",
    },
    "stale_news": {
        "task": (
            "Web-evidence: assess whether the news item at "
            "https://news.example/archive is current using source dates."
        ),
        "operation": "fetch_url",
        "url": "https://news.example/archive",
        "title": "Project launch delayed",
        "description": "Archived news article.",
        "content": (
            "Article date: 2024-02-10. The article says Project Atlas launch "
            "was delayed to late 2024. Source URL: https://news.example/archive"
        ),
        "quality_target": "Flag stale date instead of treating it as current status.",
    },
    "structured_api_page": {
        "task": (
            "Web-evidence: summarize fields returned by the public API docs at "
            "https://api.example/docs."
        ),
        "operation": "fetch_url",
        "url": "https://api.example/docs",
        "title": "Public API Reference",
        "description": "Reference for a public API.",
        "content": (
            "The docs describe JSON fields id, title, created_at, updated_at, "
            "tags, and source_url. Responses are paginated and include next_cursor. "
            "Source URL: https://api.example/docs"
        ),
        "quality_target": "List fields and pagination without inventing endpoints.",
    },
    "no_relevant_info": {
        "task": (
            "Web-evidence: find current operating hours for Example Cafe from "
            "https://example-cafe.invalid/about."
        ),
        "operation": "fetch_url",
        "url": "https://example-cafe.invalid/about",
        "title": "Example Cafe About",
        "description": "About page.",
        "content": (
            "The page contains a brand story and old photos. It does not mention "
            "operating hours, current schedule, closures, or booking links."
        ),
        "quality_target": "Mark empty or unresolved rather than inventing hours.",
    },
}


class _FakeSearxngFacility:
    """Return deterministic SearXNG-style observations for one comparison case."""

    def __init__(self, case: dict[str, Any]) -> None:
        self._case = case
        self.calls: list[dict[str, Any]] = []

    async def call_tool(self, tool_name: str, args: dict[str, Any]) -> str:
        """Record one tool call and return matching fixture evidence."""
        self.calls.append({"tool": tool_name, "args": dict(args)})
        if tool_name == "mcp-searxng__searxng_web_search":
            result = self._search_response()
            return result
        if tool_name == "mcp-searxng__web_url_read":
            result = self._read_response(args)
            return result
        result = json.dumps({"error": f"unexpected tool: {tool_name}"})
        return result

    def _search_response(self) -> str:
        """Return a stable search result list for the current case."""
        if self._case["operation"] == "search":
            items = self._case["items"]
        else:
            items = [
                {
                    "title": self._case["title"],
                    "url": self._case["url"],
                    "snippet": self._case["description"],
                    "source": "fixture",
                }
            ]
        result = json.dumps(items, ensure_ascii=False)
        return result

    def _read_response(self, args: dict[str, Any]) -> str:
        """Return a stable URL-read payload for the current case."""
        payload = {
            "url": args.get("url", self._case.get("url", "")),
            "title": self._case.get("title", ""),
            "description": self._case.get("description", ""),
            "content": self._case.get("content", ""),
        }
        result = json.dumps(payload, ensure_ascii=False)
        return result


async def _skip_if_llm_unavailable() -> None:
    """Skip comparison tests when the web-search LLM endpoint is unavailable."""
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f'{WEB_SEARCH_LLM_BASE_URL.rstrip("/")}/models'
            )
    except httpx.HTTPError:
        pytest.skip(f"LLM endpoint is unavailable: {WEB_SEARCH_LLM_BASE_URL}")

    if response.status_code >= 500:
        pytest.skip(
            f"LLM endpoint returned server error {response.status_code}: "
            f"{WEB_SEARCH_LLM_BASE_URL}"
        )


def _case_context(case: dict[str, Any]) -> dict[str, Any]:
    """Build the shared public helper context for one comparison case."""
    context = dict(_REFERENCE_CONTEXT)
    if "url" in case:
        context["source_url"] = case["url"]
    return context


async def _run_public_agent(
    agent_name: str,
    case: dict[str, Any],
    context: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Run one public web helper with real LLMs and patched SearXNG data."""
    facility = _FakeSearxngFacility(case)
    monkeypatch.setattr(web_agent2_module.mcp_manager, "call_tool", facility.call_tool)
    monkeypatch.setattr(web_agent3_tools.mcp_manager, "call_tool", facility.call_tool)

    if agent_name == "web_search_agent2":
        agent = web_agent2_module.WebSearchAgent()
    else:
        agent = web_agent3_module.WebAgent3()

    started_at = time.perf_counter()
    output = await agent.run(
        task=case["task"],
        context=context,
        max_attempts=3,
    )
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    result = {
        "elapsed_ms": elapsed_ms,
        "output": output,
        "tool_calls": list(facility.calls),
    }
    return result


def _validation_for_output(output: dict[str, Any]) -> dict[str, bool]:
    """Build structural validation flags for one public helper output."""
    cache = output.get("cache")
    validation = {
        "has_resolved_bool": isinstance(output.get("resolved"), bool),
        "has_result_string": isinstance(output.get("result"), str),
        "has_attempts_int": isinstance(output.get("attempts"), int),
        "has_cache_dict": isinstance(cache, dict),
    }
    if isinstance(cache, dict):
        validation["cache_not_enabled"] = cache.get("enabled") is False
        validation["cache_reason_present"] = isinstance(cache.get("reason"), str)
    else:
        validation["cache_not_enabled"] = False
        validation["cache_reason_present"] = False
    return validation


async def _run_comparison_case(
    case_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run one public-contract real LLM comparison and write a trace."""
    await _skip_if_llm_unavailable()
    case = _COMPARISON_CASES[case_id]
    context = _case_context(case)

    agent2_result = await _run_public_agent(
        "web_search_agent2",
        case,
        context,
        monkeypatch,
    )
    agent3_result = await _run_public_agent(
        "web_agent3",
        case,
        context,
        monkeypatch,
    )
    trace_payload = {
        "case_id": case_id,
        "model_route": "WEB_SEARCH_LLM",
        "model_name": WEB_SEARCH_LLM_MODEL,
        "comparison_boundary": "public run(task, context, max_attempts)",
        "task": case["task"],
        "context": context,
        "operation": case["operation"],
        "quality_target": case["quality_target"],
        "shared_fixture_backend": {
            "search_response": _FakeSearxngFacility(case)._search_response(),
            "read_response": _FakeSearxngFacility(case)._read_response({}),
        },
        "agent2": {
            **agent2_result,
            "validation": _validation_for_output(agent2_result["output"]),
        },
        "agent3": {
            **agent3_result,
            "validation": _validation_for_output(agent3_result["output"]),
        },
        "judgment": (
            "Manual review should compare public output usefulness, source "
            "preservation, uncertainty handling, empty-result behavior, and "
            "whether web_agent3 routing creates a regression."
        ),
    }
    trace_path = write_llm_trace(_TRACE_SUITE, case_id, trace_payload)
    logger.info(f"WEB_AGENT_PUBLIC_COMPARISON trace={trace_path} case_id={case_id}")
    logger.info(f"WEB_AGENT2 public output={agent2_result['output']}")
    logger.info(f"WEB_AGENT3 public output={agent3_result['output']}")

    assert all(trace_payload["agent2"]["validation"].values())
    assert all(trace_payload["agent3"]["validation"].values())


async def test_live_compare_official_python_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("official_python_release", monkeypatch)


async def test_live_compare_search_results_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("search_results_only", monkeypatch)


async def test_live_compare_youtube_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("youtube_description", monkeypatch)


async def test_live_compare_bilibili_description(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("bilibili_description", monkeypatch)


async def test_live_compare_nhentai_api_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("nhentai_api_docs", monkeypatch)


async def test_live_compare_redirected_docs_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("redirected_docs_page", monkeypatch)


async def test_live_compare_conflicting_snippets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("conflicting_snippets", monkeypatch)


async def test_live_compare_stale_news(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("stale_news", monkeypatch)


async def test_live_compare_structured_api_page(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("structured_api_page", monkeypatch)


async def test_live_compare_no_relevant_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _run_comparison_case("no_relevant_info", monkeypatch)
