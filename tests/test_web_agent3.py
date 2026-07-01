"""Focused deterministic tests for the RAG2 web_agent3 helper."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from kazusa_ai_chatbot.rag import web_agent3 as web_module
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3
from kazusa_ai_chatbot.rag.web_agent3 import agent as agent_module
from kazusa_ai_chatbot.rag.web_agent3 import constants as web_agent3_constants
from kazusa_ai_chatbot.rag.web_agent3 import providers as provider_module
from kazusa_ai_chatbot.rag.web_agent3 import searxng_tools as searxng_module
from kazusa_ai_chatbot.time_boundary import build_turn_clock_from_storage_utc

_WEB_AGENT3_REQUIRED_ROUTE_ENV_VARS = (
    "RELEVANCE_AGENT_LLM_BASE_URL",
    "RELEVANCE_AGENT_LLM_API_KEY",
    "RELEVANCE_AGENT_LLM_MODEL",
    "VISION_DESCRIPTOR_LLM_BASE_URL",
    "VISION_DESCRIPTOR_LLM_API_KEY",
    "VISION_DESCRIPTOR_LLM_MODEL",
    "MSG_DECONTEXTUALIZER_LLM_BASE_URL",
    "MSG_DECONTEXTUALIZER_LLM_API_KEY",
    "MSG_DECONTEXTUALIZER_LLM_MODEL",
    "RAG_PLANNER_LLM_BASE_URL",
    "RAG_PLANNER_LLM_API_KEY",
    "RAG_PLANNER_LLM_MODEL",
    "RAG_SUBAGENT_LLM_BASE_URL",
    "RAG_SUBAGENT_LLM_API_KEY",
    "RAG_SUBAGENT_LLM_MODEL",
    "WEB_SEARCH_LLM_BASE_URL",
    "WEB_SEARCH_LLM_API_KEY",
    "WEB_SEARCH_LLM_MODEL",
    "COGNITION_LLM_BASE_URL",
    "COGNITION_LLM_API_KEY",
    "COGNITION_LLM_MODEL",
    "BOUNDARY_CORE_LLM_BASE_URL",
    "BOUNDARY_CORE_LLM_API_KEY",
    "BOUNDARY_CORE_LLM_MODEL",
    "DIALOG_GENERATOR_LLM_BASE_URL",
    "DIALOG_GENERATOR_LLM_API_KEY",
    "DIALOG_GENERATOR_LLM_MODEL",
    "CONSOLIDATION_LLM_BASE_URL",
    "CONSOLIDATION_LLM_API_KEY",
    "CONSOLIDATION_LLM_MODEL",
    "JSON_REPAIR_LLM_BASE_URL",
    "JSON_REPAIR_LLM_API_KEY",
    "JSON_REPAIR_LLM_MODEL",
    "BACKGROUND_ARTIFACT_LLM_BASE_URL",
    "BACKGROUND_ARTIFACT_LLM_API_KEY",
    "BACKGROUND_ARTIFACT_LLM_MODEL",
)


def _web_agent3_subprocess_env(
    *,
    searxng_url: str | None,
    nhentai_token: str | None,
) -> dict[str, str]:
    """Build an import environment for web_agent3 availability tests."""
    env = dict(os.environ)
    python_path = env.get("PYTHONPATH", "")
    src_path = os.path.abspath("src")
    env["PYTHONPATH"] = (
        src_path if not python_path else f"{src_path}{os.pathsep}{python_path}"
    )
    env["PYTHON_DOTENV_DISABLED"] = "1"
    for name in _WEB_AGENT3_REQUIRED_ROUTE_ENV_VARS:
        env[name] = "configured"
    env["EMBEDDING_BASE_URL"] = "configured"
    env["EMBEDDING_API_KEY"] = "configured"
    env["EMBEDDING_MODEL"] = "configured"
    env["CHARACTER_GLOBAL_USER_ID"] = "character-global"

    if searxng_url is None:
        env.pop("SEARXNG_URL", None)
    else:
        env["SEARXNG_URL"] = searxng_url

    if nhentai_token is None:
        env.pop("NHENTAI_TOKEN", None)
    else:
        env["NHENTAI_TOKEN"] = nhentai_token

    return env


def _read_web_agent3_source_state(
    tmp_path: Path,
    *,
    searxng_url: str | None = None,
    nhentai_token: str | None = None,
) -> dict[str, object]:
    """Import web_agent3 in a subprocess and return source prompt state."""
    env = _web_agent3_subprocess_env(
        searxng_url=searxng_url,
        nhentai_token=nhentai_token,
    )
    script = (
        "import json; "
        "from kazusa_ai_chatbot.rag.web_agent3 import agent; "
        "from kazusa_ai_chatbot.rag.web_agent3 import subagent; "
        "payload = {"
        "'names': list(subagent._SUBAGENT_NAMES), "
        "'descriptions': list(subagent._SUBAGENT_DESCRIPTIONS), "
        "'actions': subagent._SUBAGENT_SUPPORTED_ACTIONS, "
        "'source_text': agent._WEB_AGENT3_SOURCE_TOOLS_TEXT, "
        "'prompt': agent._WEB_AGENT3_GENERATOR_PROMPT"
        "}; "
        "print(json.dumps(payload))"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    return payload


class _FakeHTTPResponse:
    """Small successful HTTP response fixture for direct web tests."""

    def __init__(
        self,
        *,
        text: str = "",
        headers: dict[str, str] | None = None,
        json_payload: dict | None = None,
        chunks: list[bytes] | None = None,
        status_code: int = 200,
        url: str = "https://example.test/page",
        cookies: httpx.Cookies | None = None,
    ) -> None:
        self.text = text
        self.content = text.encode("utf-8")
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}
        self._json_payload = json_payload
        self._chunks = chunks
        self.status_code = status_code
        self.url = url
        self.cookies = cookies or httpx.Cookies()

    def raise_for_status(self) -> None:
        if self.status_code < 400:
            return None

        request = httpx.Request("GET", self.url)
        response = httpx.Response(
            self.status_code,
            headers=self.headers,
            request=request,
        )
        response.raise_for_status()

    def json(self) -> dict:
        if self._json_payload is None:
            raise ValueError("response is not JSON")
        return_value = self._json_payload
        return return_value

    async def aiter_bytes(self):
        """Yield configured response chunks for streaming URL-reader tests."""
        if self._chunks is None:
            yield self.content
            return

        for chunk in self._chunks:
            yield chunk


class _FakeHTTPStream:
    """Async context manager wrapping a fake streaming response."""

    def __init__(self, response: _FakeHTTPResponse) -> None:
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc, tb):
        return None


def test_web_agent3_is_subpackage_with_icd() -> None:
    """web_agent3 should be an importable package with a local ICD."""
    package_path = Path(web_module.__file__).parent

    assert package_path.name == "web_agent3"
    assert (package_path / "README.md").is_file()


@pytest.mark.asyncio
async def test_web_agent3_search_reports_unavailable_without_searxng_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should degrade cleanly when no direct SearXNG URL is set."""
    direct_searxng = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.direct_searxng"
    )
    monkeypatch.setattr(direct_searxng, "SEARXNG_URL", "")

    result = await searxng_module.web_search.ainvoke({"query": "test query"})

    assert result == (
        "Error: SearXNG search unavailable: SEARXNG_URL is not configured."
    )


@pytest.mark.asyncio
async def test_web_agent3_search_calls_direct_searxng_json_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should call SearXNG /search directly with JSON params."""
    direct_searxng = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.direct_searxng"
    )
    calls: list[dict[str, object]] = []

    class FakeSearchClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, *, params: dict, headers: dict | None = None):
            calls.append({
                "url": url,
                "params": params,
                "headers": headers,
                "kwargs": self.kwargs,
            })
            response = _FakeHTTPResponse(json_payload={
                "results": [
                    {
                        "title": "First result",
                        "url": "https://example.test/first",
                        "content": "Alpha snippet",
                        "engine": "fixture",
                        "score": 1.5,
                    }
                ],
            })
            return response

    monkeypatch.setattr(direct_searxng, "SEARXNG_URL", "http://search.test")
    monkeypatch.setattr(direct_searxng, "SEARXNG_SEARCH_TIMEOUT_SECONDS", 12.0)
    monkeypatch.setattr(direct_searxng, "SEARXNG_SEARCH_RESULT_LIMIT", 10)
    monkeypatch.setattr(direct_searxng.httpx, "AsyncClient", FakeSearchClient)

    result = await searxng_module.web_search.ainvoke({
        "query": "direct search",
        "pageno": 2,
        "time_range": "month",
        "language": "en",
    })

    assert calls == [{
        "url": "http://search.test/search",
        "params": {
            "q": "direct search",
            "format": "json",
            "pageno": 2,
            "time_range": "month",
            "language": "en",
            "safesearch": 0,
        },
        "headers": None,
        "kwargs": {"timeout": 12.0},
    }]
    assert "Title: First result" in result
    assert "URL: https://example.test/first" in result
    assert "Snippet: Alpha snippet" in result
    assert "Engine: fixture" in result
    assert "Score: 1.5" in result


@pytest.mark.asyncio
async def test_web_agent3_search_omits_empty_optional_searxng_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should not send blank optional params to SearXNG."""
    direct_searxng = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.direct_searxng"
    )
    calls: list[dict[str, object]] = []

    class FakeSearchClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, *, params: dict, headers: dict | None = None):
            calls.append({
                "url": url,
                "params": params,
                "headers": headers,
                "kwargs": self.kwargs,
            })
            response = _FakeHTTPResponse(json_payload={"results": []})
            return response

    monkeypatch.setattr(direct_searxng, "SEARXNG_URL", "http://search.test")
    monkeypatch.setattr(direct_searxng, "SEARXNG_SEARCH_TIMEOUT_SECONDS", 12.0)
    monkeypatch.setattr(direct_searxng.httpx, "AsyncClient", FakeSearchClient)

    result = await searxng_module.web_search.ainvoke({
        "query": "direct search",
        "pageno": 1,
        "time_range": "",
        "language": "   ",
    })

    assert calls == [{
        "url": "http://search.test/search",
        "params": {
            "q": "direct search",
            "format": "json",
            "pageno": 1,
            "safesearch": 0,
        },
        "headers": None,
        "kwargs": {"timeout": 12.0},
    }]
    assert result == "No results found."


@pytest.mark.asyncio
async def test_web_agent3_search_formats_bounded_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should format only the configured number of result rows."""
    direct_searxng = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.direct_searxng"
    )

    class FakeSearchClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        async def get(self, url: str, *, params: dict, headers: dict | None = None):
            response = _FakeHTTPResponse(json_payload={
                "results": [
                    {
                        "title": "Kept",
                        "url": "https://example.test/kept",
                        "content": "Kept snippet",
                    },
                    {
                        "title": "Dropped",
                        "url": "https://example.test/dropped",
                        "content": "Dropped snippet",
                    },
                ],
            })
            return response

    monkeypatch.setattr(direct_searxng, "SEARXNG_URL", "http://search.test")
    monkeypatch.setattr(direct_searxng, "SEARXNG_SEARCH_RESULT_LIMIT", 1)
    monkeypatch.setattr(direct_searxng.httpx, "AsyncClient", FakeSearchClient)

    result = await searxng_module.web_search.ainvoke({"query": "bounded"})

    assert "Title: Kept" in result
    assert "https://example.test/kept" in result
    assert "Dropped" not in result
    assert "https://example.test/dropped" not in result


@pytest.mark.asyncio
async def test_web_agent3_url_read_sends_configured_browser_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should send configured browser-like headers."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    calls: list[dict[str, object]] = []

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            calls.append({"url": url, "headers": headers, "kwargs": self.kwargs})
            response = _FakeHTTPResponse(text="<html><body><p>Page body</p></body></html>")
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader, "WEB_URL_READER_USER_AGENT", "TestBrowser/1.0")
    monkeypatch.setattr(url_reader, "WEB_URL_READER_ACCEPT_LANGUAGE", "ja,en;q=0.8")
    monkeypatch.setattr(url_reader, "WEB_URL_READ_TIMEOUT_SECONDS", 9.0)
    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/page",
    })

    assert calls[0]["url"] == "https://example.test/page"
    headers = calls[0]["headers"]
    assert headers["User-Agent"] == "TestBrowser/1.0"
    assert headers["Accept-Language"] == "ja,en;q=0.8"
    assert "text/html" in headers["Accept"]
    assert "image/webp" in headers["Accept"]
    assert headers["Accept-Encoding"].startswith("gzip, deflate")
    assert headers["DNT"] == "1"
    assert headers["Upgrade-Insecure-Requests"] == "1"
    assert headers["Sec-Fetch-Dest"] == "document"
    assert headers["Sec-Fetch-Mode"] == "navigate"
    assert headers["Sec-Fetch-Site"] == "none"
    assert headers["Sec-Fetch-User"] == "?1"
    assert headers["Referer"] == "https://example.test/"
    assert calls[0]["kwargs"]["timeout"] == 9.0
    assert "Page body" in result


@pytest.mark.asyncio
async def test_web_agent3_url_read_accept_encoding_uses_available_decoders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should only advertise locally supported compression."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    calls: list[dict[str, str]] = []

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            calls.append(headers)
            response = _FakeHTTPResponse(
                text="<html><body><p>Compressed page</p></body></html>",
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader, "_BROTLI_AVAILABLE", True)
    monkeypatch.setattr(url_reader, "_ZSTD_AVAILABLE", True)
    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/compressed",
    })

    assert calls[0]["Accept-Encoding"] == "gzip, deflate, br, zstd"
    assert "Compressed page" in result


@pytest.mark.asyncio
async def test_web_agent3_url_read_accept_encoding_omits_missing_decoders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should not advertise optional missing decoders."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    calls: list[dict[str, str]] = []

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            calls.append(headers)
            response = _FakeHTTPResponse(
                text="<html><body><p>Plain compressed page</p></body></html>",
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader, "_BROTLI_AVAILABLE", False)
    monkeypatch.setattr(url_reader, "_ZSTD_AVAILABLE", False)
    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/compressed",
    })

    assert calls[0]["Accept-Encoding"] == "gzip, deflate"
    assert "Plain compressed page" in result


@pytest.mark.asyncio
async def test_web_agent3_url_read_reuses_process_memory_cookies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should keep HTTP cookies in process memory."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    calls: list[dict[str, object]] = []
    response_cookies = httpx.Cookies()
    response_cookies.set("sessionid", "abc", domain="example.test")
    responses = [
        _FakeHTTPResponse(
            text="<html><body><p>First page</p></body></html>",
            cookies=response_cookies,
        ),
        _FakeHTTPResponse(text="<html><body><p>Second page</p></body></html>"),
    ]

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            calls.append({"url": url, "headers": headers, "kwargs": self.kwargs})
            response = responses.pop(0)
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader, "_COOKIE_JAR", httpx.Cookies())
    monkeypatch.setattr(url_reader, "_COOKIE_LOCK", asyncio.Lock())
    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    first_result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/first",
    })
    second_result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/second",
    })

    first_kwargs = calls[0]["kwargs"]
    second_kwargs = calls[1]["kwargs"]
    first_cookies = first_kwargs["cookies"]
    second_cookies = second_kwargs["cookies"]
    assert first_cookies.get("sessionid") is None
    assert second_cookies.get("sessionid") == "abc"
    assert "First page" in first_result
    assert "Second page" in second_result


@pytest.mark.asyncio
async def test_web_agent3_url_read_reuses_redirect_cookies_from_client_jar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should keep cookies collected during redirects."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    calls: list[dict[str, object]] = []
    client_count = 0

    class FakeURLClient:
        def __init__(self, **kwargs):
            nonlocal client_count

            client_count += 1
            self.kwargs = kwargs
            self.cookies = httpx.Cookies()
            if client_count == 1:
                self.cookies.set(
                    "redirectid",
                    "xyz",
                    domain="example.test",
                )

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            calls.append({"url": url, "headers": headers, "kwargs": self.kwargs})
            response = _FakeHTTPResponse(
                text="<html><body><p>Redirect page</p></body></html>",
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader, "_COOKIE_JAR", httpx.Cookies())
    monkeypatch.setattr(url_reader, "_COOKIE_LOCK", asyncio.Lock())
    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/redirect",
    })
    await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/next",
    })

    second_kwargs = calls[1]["kwargs"]
    second_cookies = second_kwargs["cookies"]
    assert second_cookies.get("redirectid") == "xyz"


@pytest.mark.asyncio
async def test_web_agent3_url_read_referer_omits_url_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should not put URL userinfo into synthetic Referer."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    calls: list[dict[str, str]] = []

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            calls.append(headers)
            response = _FakeHTTPResponse(
                text="<html><body><p>Credential URL page</p></body></html>",
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://user:pass@example.test/path",
    })

    assert calls[0]["Referer"] == "https://example.test/"
    assert "Credential URL page" in result


@pytest.mark.asyncio
async def test_web_agent3_url_read_reports_anti_bot_challenge_before_http_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should identify common HTTP anti-bot challenge pages."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            response = _FakeHTTPResponse(
                text=(
                    "<html><title>Just a moment...</title>"
                    "<body>Checking your browser before accessing.</body></html>"
                ),
                headers={
                    "content-type": "text/html; charset=utf-8",
                    "server": "cloudflare",
                    "cf-ray": "fixture",
                },
                status_code=403,
                url=url,
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/protected",
    })

    assert result == (
        "Error: URL read blocked by anti-bot challenge: cloudflare (HTTP 403)"
    )


@pytest.mark.asyncio
async def test_web_agent3_url_read_does_not_mask_ok_pages_with_marker_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should not treat ordinary HTTP 200 text as a challenge."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            response = _FakeHTTPResponse(
                text=(
                    "<html><body><p>This documentation says Just a moment... "
                    "and describes checking your browser text.</p></body></html>"
                ),
                status_code=200,
                url=url,
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/docs",
    })

    assert "This documentation says Just a moment..." in result
    assert not result.startswith("Error: URL read blocked by anti-bot challenge")


@pytest.mark.asyncio
async def test_web_agent3_url_read_preserves_non_challenge_http_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should keep ordinary HTTP errors separate from bot checks."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            response = _FakeHTTPResponse(
                text="<html><body><p>Forbidden.</p></body></html>",
                status_code=403,
                url=url,
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/forbidden",
    })

    assert result.startswith("Error: URL read HTTP error:")
    assert "403 Forbidden" in result


@pytest.mark.asyncio
async def test_web_agent3_url_read_accepts_local_http_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should allow local HTTP resources by default."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    calls: list[str] = []

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            calls.append(url)
            response = _FakeHTTPResponse(text="<html><body><p>Local page</p></body></html>")
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "http://127.0.0.1:8765/status",
    })

    assert calls == ["http://127.0.0.1:8765/status"]
    assert "Local page" in result


@pytest.mark.asyncio
async def test_web_agent3_url_read_rejects_non_http_scheme() -> None:
    """web_url_read should reject local filesystem schemes."""
    result = await searxng_module.web_url_read.ainvoke({
        "url": "file:///etc/passwd",
    })

    assert result == "Error: unsupported URL scheme: file"


@pytest.mark.asyncio
async def test_web_agent3_url_read_returns_error_for_malformed_http_url() -> None:
    """web_url_read should not raise when URL parsing rejects HTTP text."""
    result = await searxng_module.web_url_read.ainvoke({
        "url": "http://[::1",
    })

    assert result.startswith("Error: invalid URL:")


@pytest.mark.asyncio
async def test_web_agent3_url_read_returns_error_for_httpx_invalid_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should bound URL validation failures from httpx."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            raise httpx.InvalidURL("invalid URL from client")

    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/page",
    })

    assert result == "Error: invalid URL: invalid URL from client"


@pytest.mark.asyncio
async def test_web_agent3_url_read_stops_stream_when_response_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should stop reading once the byte cap is exceeded."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    yielded_chunks: list[bytes] = []

    class StreamingResponse(_FakeHTTPResponse):
        async def aiter_bytes(self):
            for chunk in (b"abcd", b"efgh", b"ignored"):
                yielded_chunks.append(chunk)
                yield chunk

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            response = StreamingResponse(
                text="",
                headers={"content-type": "text/plain; charset=utf-8"},
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader, "WEB_URL_READ_MAX_BYTES", 5)
    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/large",
    })

    assert result == (
        "Error: response too large: exceeds WEB_URL_READ_MAX_BYTES=5"
    )
    assert yielded_chunks == [b"abcd", b"efgh"]


@pytest.mark.asyncio
async def test_web_agent3_url_read_extracts_headings_sections_paragraphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_url_read should expose headings, section slices, and paragraphs."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    html = """
    <html>
      <head><title>Fixture Doc</title><script>hidden()</script></head>
      <body>
        <h1>Guide</h1>
        <p>Intro paragraph.</p>
        <h2>Usage</h2>
        <p>First usage paragraph.</p>
        <p>Second usage paragraph.</p>
        <h2>Other</h2>
        <p>Other text.</p>
      </body>
    </html>
    """

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            response = _FakeHTTPResponse(text=html)
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    headings = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/guide",
        "readHeadings": True,
    })
    section = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/guide",
        "section": "Usage",
    })
    paragraph = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/guide",
        "section": "Usage",
        "paragraphRange": "2",
    })

    assert "Guide" in headings
    assert "Usage" in headings
    assert "Other" in headings
    assert "hidden" not in headings
    assert "First usage paragraph." in section
    assert "Second usage paragraph." in section
    assert "Other text." not in section
    assert "Second usage paragraph." in paragraph
    assert "First usage paragraph." not in paragraph


@pytest.mark.asyncio
async def test_web_agent3_url_read_caps_zero_max_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A zero maxLength should use the configured max char cap."""
    url_reader = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.url_reader"
    )
    long_text = "abcdefghijklmnopqrstuvwxyz"

    class FakeURLClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return None

        def stream(self, method: str, url: str, *, headers: dict):
            assert method == "GET"
            response = _FakeHTTPResponse(
                text=long_text,
                headers={"content-type": "text/plain; charset=utf-8"},
            )
            stream = _FakeHTTPStream(response)
            return stream

    monkeypatch.setattr(url_reader, "WEB_URL_READ_MAX_CHARS", 12)
    monkeypatch.setattr(url_reader.httpx, "AsyncClient", FakeURLClient)

    result = await searxng_module.web_url_read.ainvoke({
        "url": "https://example.test/text",
        "maxLength": 0,
    })

    assert result == "abcdefghijkl"


def test_web_agent3_router_output_parsing_is_minimal() -> None:
    """Router parsing should keep only action, source, and query semantics."""
    decision = web_module._normalize_router_decision(
        {
            "action": "read",
            "source": "nhentai",
            "query": "652244",
            "reason": "ignored",
            "api_params": {"id": 652244},
        },
        fallback_query="fallback search",
        valid_sources=("web_read", "nhentai"),
        source_actions={
            "web_read": ("read",),
            "nhentai": ("read", "search"),
        },
    )

    assert decision == web_module._RouterDecision(
        action="read",
        source="nhentai",
        query="652244",
    )


def test_web_agent3_router_prompt_omits_input_format_headers() -> None:
    """Router prompt should not use the retired input-format headers."""
    forbidden_headers = [
        "# " + "\u8f93\u5165\u683c\u5f0f",
        "# " + "Input Format",
    ]

    for header in forbidden_headers:
        assert header not in agent_module._WEB_AGENT3_GENERATOR_PROMPT


def test_web_agent3_router_prompt_uses_project_prompt_style() -> None:
    """Router prompt should use the static project prompt style."""
    prompt = agent_module._WEB_AGENT3_GENERATOR_PROMPT

    assert "# 来源原则" in prompt
    assert "# 审计步骤" in prompt
    assert "# 输出格式" in prompt
    assert "# 输出契约" not in prompt
    assert "source adapter roster" not in prompt
    assert agent_module._WEB_AGENT3_SOURCE_TOOLS_TEXT in prompt
    assert '"source": "string"' in prompt
    assert "不要因为 `reference_time` 自动把当前日期" in prompt
    assert "移除日期/年份和堆叠约束" in prompt
    assert "多个来源类别、发布轨道、证据立场或资料类型" in prompt
    assert "从任务和上下文合理推断的 URL" in prompt
    assert "没有任何 `read` 尝试" in agent_module._WEB_AGENT3_EVALUATOR_PROMPT
    hardcoded_source_schema = '"source": "{}"'.format(
        "|".join(["generic", "bilibili", "youtube", "nhentai"])
    )
    assert hardcoded_source_schema not in prompt


def test_web_agent3_router_uses_subagent_generation_rules() -> None:
    """Router query guidance should come from source subagent descriptions."""
    web_read_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_read"
    )

    assert "read:" in web_read_subagent.DESCRIPTION
    assert "HTTP(S) URL" in web_read_subagent.DESCRIPTION
    assert "正文读取才是强证据" in web_read_subagent.DESCRIPTION
    assert web_read_subagent.DESCRIPTION in agent_module._WEB_AGENT3_SOURCE_TOOLS_TEXT
    assert "遵循所选来源描述" in agent_module._WEB_AGENT3_GENERATOR_PROMPT


def test_web_agent3_router_source_text_omits_execution_details() -> None:
    """Router source descriptions should expose capability, not implementation."""
    source_text = agent_module._WEB_AGENT3_SOURCE_TOOLS_TEXT
    forbidden_terms = [
        'SearXNG',
        '当前占位',
        '不会调用',
        '回退',
        'fallback',
    ]

    for forbidden_term in forbidden_terms:
        assert forbidden_term not in source_text


def test_web_agent3_source_subagents_are_discovered_from_subagent_package() -> None:
    """Source subagents should be discovered from enabled per-source modules."""
    package_path = Path(web_module.__file__).parent
    subagent_path = package_path / "subagent"
    retired_source_file = package_path / "source_subagents.py"

    assert subagent_path.is_dir()
    assert not retired_source_file.exists()

    source_module = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent"
    )
    enabled_sources = set(source_module._SUBAGENTS)
    allowed_sources = {"web_read", "web_search", "nhentai"}

    assert Path(source_module.__file__).name == "__init__.py"
    assert "web_read" in enabled_sources
    assert enabled_sources <= allowed_sources
    assert set(source_module._SUBAGENT_DESCRIPTIONS) == enabled_sources
    assert set(source_module._SUBAGENT_SUPPORTED_ACTIONS) == enabled_sources
    assert list(source_module._SUBAGENT_DESCRIPTIONS) == sorted(enabled_sources)

    for source in enabled_sources:
        module_path = subagent_path / f"{source}.py"
        source_subagent = importlib.import_module(
            f"kazusa_ai_chatbot.rag.web_agent3.subagent.{source}"
        )

        assert module_path.is_file()
        assert source_subagent.SOURCE == source
        assert source_subagent.DESCRIPTION
        assert source_subagent.SUPPORTED_ACTIONS
        assert callable(source_subagent.execute)

    assert not hasattr(provider_module, "_SOURCE_SUBAGENTS")
    assert not hasattr(provider_module, "_SOURCE_ADAPTER_DESCRIPTIONS")
    assert not hasattr(provider_module, "_SOURCE_ADAPTERS")


def test_web_agent3_source_discovery_registers_only_web_read_without_optional_config(
    tmp_path: Path,
) -> None:
    """Only URL read should be available when optional source config is absent."""
    payload = _read_web_agent3_source_state(tmp_path)

    assert payload["names"] == ["web_read"]
    assert payload["descriptions"] == ["web_read"]
    assert payload["actions"] == {"web_read": ["read"]}


def test_web_agent3_source_discovery_registers_web_search_when_searxng_configured(
    tmp_path: Path,
) -> None:
    """SearXNG config should enable the web_search source."""
    payload = _read_web_agent3_source_state(
        tmp_path,
        searxng_url="http://search.test:8080",
    )

    assert set(payload["names"]) == {"web_read", "web_search"}
    assert payload["actions"]["web_read"] == ["read"]
    assert payload["actions"]["web_search"] == ["search"]


def test_web_agent3_source_discovery_registers_nhentai_when_token_configured(
    tmp_path: Path,
) -> None:
    """nHentai token config should enable the nhentai source."""
    payload = _read_web_agent3_source_state(
        tmp_path,
        nhentai_token="secret-token",
    )

    assert set(payload["names"]) == {"web_read", "nhentai"}
    assert payload["actions"]["web_read"] == ["read"]
    assert payload["actions"]["nhentai"] == ["read", "search"]


def test_web_agent3_source_discovery_registers_all_configured_sources(
    tmp_path: Path,
) -> None:
    """All final sources should be available when all optional config exists."""
    payload = _read_web_agent3_source_state(
        tmp_path,
        searxng_url="http://search.test:8080",
        nhentai_token="secret-token",
    )

    assert set(payload["names"]) == {"web_read", "web_search", "nhentai"}
    assert payload["actions"] == {
        "nhentai": ["read", "search"],
        "web_read": ["read"],
        "web_search": ["search"],
    }


def test_web_agent3_router_prompt_lists_enabled_sources_only(
    tmp_path: Path,
) -> None:
    """Router prompt source text should expose only enabled final sources."""
    no_optional_payload = _read_web_agent3_source_state(tmp_path)
    all_enabled_payload = _read_web_agent3_source_state(
        tmp_path,
        searxng_url="http://search.test:8080",
        nhentai_token="secret-token",
    )

    no_optional_prompt = str(no_optional_payload["prompt"])
    all_enabled_prompt = str(all_enabled_payload["prompt"])
    assert "- web_read:" in no_optional_prompt
    assert "- web_search:" not in no_optional_prompt
    assert "- nhentai:" not in no_optional_prompt

    assert "- web_read:" in all_enabled_prompt
    assert "- web_search:" in all_enabled_prompt
    assert "- nhentai:" in all_enabled_prompt
    for removed_source in ("generic", "bilibili", "youtube"):
        assert removed_source not in all_enabled_prompt


def test_web_agent3_removed_source_modules_are_absent() -> None:
    """Removed source module files should not remain importable."""
    subagent_path = Path(web_module.__file__).parent / "subagent"

    for removed_source in ("generic", "bilibili", "youtube"):
        module_path = subagent_path / f"{removed_source}.py"
        assert not module_path.exists()


@pytest.mark.asyncio
async def test_web_agent3_generator_outputs_router_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generator should parse a strict action/source/query router decision."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content='{"action": "search", "source": "web_search", "query": "demo"}',
        )),
    )
    monkeypatch.setattr(agent_module, "_generator_llm", fake_llm)
    monkeypatch.setattr(agent_module, "_SUBAGENT_NAMES", ("web_read", "web_search"))
    monkeypatch.setattr(
        agent_module,
        "_SUBAGENT_SUPPORTED_ACTIONS",
        {"web_read": ("read",), "web_search": ("search",)},
    )
    state = {
        "task": "Find a web demo.",
        "context": {"platform": "debug"},
        "messages": [HumanMessage(content="start")],
        "observations": [{"action": "search", "source": "web_search"}],
        "evaluator_feedback": "read a web result next",
        "prompt_timestamp": "2026-05-25 21:30 (Monday)",
    }

    update = await agent_module._tool_call_generator(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    system_prompt = messages[0].content
    payload = json.loads(messages[1].content)
    assert "2026-05-25" not in system_prompt
    assert payload["reference_time"] == "2026-05-25 21:30 (Monday)"
    assert payload["evaluator_feedback"] == "read a web result next"
    assert update["router_decision"] == {
        "action": "search",
        "source": "web_search",
        "query": "demo",
    }


@pytest.mark.asyncio
async def test_web_agent3_web_search_receives_query_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should pass router query directly to the search tool."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )
    fake_search = SimpleNamespace(ainvoke=AsyncMock(return_value="search body"))
    monkeypatch.setitem(
        provider_module._source_subagent_package._SUBAGENTS,
        "web_search",
        web_search_subagent,
    )
    monkeypatch.setattr(web_search_subagent.searxng_tools, "web_search", fake_search)
    decision = web_module._RouterDecision(
        action="search",
        source="web_search",
        query="local tool router demo web agent architecture",
    )

    result = await web_module._execute_source_decision(decision)

    fake_search.ainvoke.assert_awaited_once_with({
        "query": "local tool router demo web agent architecture",
    })
    assert result == "search body"


@pytest.mark.asyncio
async def test_web_agent3_web_search_expands_dense_query_inside_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should expand dense semantic input inside the source."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )
    generated_attempts = [
        {
            "query": "alpha service setup time",
            "purpose": "Find setup-time evidence for alpha service.",
        },
        {
            "query": "beta service setup time",
            "purpose": "Find setup-time evidence for beta service.",
        },
        {
            "query": "alpha beta service platform support",
            "purpose": "Find platform-support evidence for both services.",
        },
    ]
    attempt_generator = AsyncMock(return_value=generated_attempts)
    monkeypatch.setattr(
        web_search_subagent,
        "_generate_search_attempts",
        attempt_generator,
    )
    search_calls: list[dict[str, str]] = []

    async def fake_search(payload: dict[str, str]) -> str:
        search_calls.append(payload)
        query = payload["query"]
        return_value = (
            f"Title: Result for {query}\n"
            f"URL: https://example.test/{len(search_calls)}\n"
            f"Snippet: Evidence snippet for {query}"
        )
        return return_value

    fake_search_tool = SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_search))
    monkeypatch.setattr(
        web_search_subagent.searxng_tools,
        "web_search",
        fake_search_tool,
    )
    decision = web_module._RouterDecision(
        action="search",
        source="web_search",
        query=(
            "Compare alpha service with beta service for setup time, monthly "
            "price, and platform support."
        ),
    )

    result = await web_search_subagent.execute(decision)

    attempt_generator.assert_awaited_once_with(decision.query)
    assert search_calls == [
        {"query": "alpha service setup time"},
        {"query": "beta service setup time"},
        {"query": "alpha beta service platform support"},
    ]
    assert "Search attempts:" in result
    assert "Query: alpha service setup time" in result
    assert "Purpose: Find setup-time evidence for alpha service." in result
    assert "Key evidence:" in result
    assert "Missing or weak coverage:" in result
    assert "Recommended narrower search focus:" in result
    for forbidden_text in (
        "schema_version",
        "cache_name",
        "trace_id",
        "SearXNG params",
    ):
        assert forbidden_text not in result


@pytest.mark.asyncio
async def test_web_agent3_web_search_preserves_simple_query_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should keep a simple direct query as one source call."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )
    attempt_generator = AsyncMock(side_effect=AssertionError(
        "simple query should not use search attempt expansion"
    ))
    monkeypatch.setattr(
        web_search_subagent,
        "_generate_search_attempts",
        attempt_generator,
        raising=False,
    )
    fake_search = SimpleNamespace(ainvoke=AsyncMock(return_value="search body"))
    monkeypatch.setattr(web_search_subagent.searxng_tools, "web_search", fake_search)
    decision = web_module._RouterDecision(
        action="search",
        source="web_search",
        query="local tool router demo web agent architecture",
    )

    result = await web_search_subagent.execute(decision)

    fake_search.ainvoke.assert_awaited_once_with({
        "query": "local tool router demo web agent architecture",
    })
    assert attempt_generator.await_count == 0
    assert result == "search body"


@pytest.mark.asyncio
async def test_web_agent3_web_search_expansion_dedupes_and_caps_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_search should validate, dedupe, and cap generated attempts."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )
    generated_attempts = [
        {"query": "   ", "purpose": "ignored empty"},
        {"query": "alpha support status", "purpose": "first useful target"},
        {"query": "alpha support status", "purpose": "duplicate target"},
        {"query": " beta support status ", "purpose": "second useful target"},
        {"query": "gamma support status", "purpose": "over cap target"},
    ]
    monkeypatch.setattr(web_agent3_constants, "MAX_SEARCH_ATTEMPTS", 2)
    monkeypatch.setattr(
        web_search_subagent,
        "_generate_search_attempts",
        AsyncMock(return_value=generated_attempts),
    )
    search_calls: list[dict[str, str]] = []

    async def fake_search(payload: dict[str, str]) -> str:
        search_calls.append(payload)
        if payload["query"] == "alpha support status":
            return "No results found."
        return "Title: Beta support\nURL: https://example.test/beta"

    fake_search_tool = SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_search))
    monkeypatch.setattr(
        web_search_subagent.searxng_tools,
        "web_search",
        fake_search_tool,
    )
    decision = web_module._RouterDecision(
        action="search",
        source="web_search",
        query="Compare alpha, beta, and gamma support status.",
    )

    result = await web_search_subagent.execute(decision)

    assert search_calls == [
        {"query": "alpha support status"},
        {"query": "beta support status"},
    ]
    assert "Query: gamma support status" not in result
    assert "Result: no useful result" in result
    assert "Result: source evidence returned" in result


@pytest.mark.asyncio
async def test_web_agent3_web_search_expansion_rejects_metadata_attempt_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generated attempt rows must stay semantic before search or projection."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )
    generated_attempts = [
        {
            "query": {"schema_version": "leaked"},
            "purpose": "malformed query should be ignored",
        },
        {
            "query": "schema_version trace_id provider params",
            "purpose": "metadata-shaped text should be ignored",
        },
        {
            "query": "cache hit true provider_params searxng_params",
            "purpose": "metadata-shaped underscore text should be ignored",
        },
        {
            "query": "node_id task_1 source_node_id task_2",
            "purpose": "graph metadata should be ignored",
        },
        {
            "query": "attempt_index stage_name route_name",
            "purpose": "stage metadata should be ignored",
        },
        {
            "query": "valid focused public source search",
            "purpose": {"trace_id": "leaked"},
        },
    ]
    monkeypatch.setattr(
        web_search_subagent,
        "_generate_search_attempts",
        AsyncMock(return_value=generated_attempts),
    )
    search_calls: list[dict[str, str]] = []

    async def fake_search(payload: dict[str, str]) -> str:
        search_calls.append(payload)
        return "Title: Valid source\nURL: https://example.test/source"

    fake_search_tool = SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_search))
    monkeypatch.setattr(
        web_search_subagent.searxng_tools,
        "web_search",
        fake_search_tool,
    )
    decision = web_module._RouterDecision(
        action="search",
        source="web_search",
        query="Compare several public facts across sources and versions.",
    )

    result = await web_search_subagent.execute(decision)

    assert search_calls == [{"query": "valid focused public source search"}]
    assert "Search for one focused part of the request." in result
    for forbidden_text in (
        "schema_version",
        "trace_id",
        "cache hit",
        "provider params",
        "provider_params",
        "searxng_params",
        "node_id",
        "source_node_id",
        "attempt_index",
        "stage_name",
        "route_name",
        "{'trace_id'",
        "{'schema_version'",
    ):
        assert forbidden_text not in result


@pytest.mark.asyncio
async def test_web_agent3_web_search_reports_invalid_expansion_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fallback to the original dense query must remain visible downstream."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )
    monkeypatch.setattr(
        web_search_subagent,
        "_generate_search_attempts",
        AsyncMock(return_value=[
            {"query": {"trace_id": "bad"}, "purpose": "bad"},
            {"query": "schema_version cache_name", "purpose": "bad"},
        ]),
    )
    search_calls: list[dict[str, str]] = []

    async def fake_search(payload: dict[str, str]) -> str:
        search_calls.append(payload)
        return "Title: Broad source\nURL: https://example.test/broad"

    fake_search_tool = SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_search))
    monkeypatch.setattr(
        web_search_subagent.searxng_tools,
        "web_search",
        fake_search_tool,
    )
    dense_query = "Compare alpha, beta, and gamma across current docs."
    decision = web_module._RouterDecision(
        action="search",
        source="web_search",
        query=dense_query,
    )

    result = await web_search_subagent.execute(decision)

    assert search_calls == [{"query": dense_query}]
    assert "Search-attempt expansion produced no valid focused queries" in result
    assert "No tool-level missing result was observed" not in result
    assert "schema_version" not in result
    assert "cache_name" not in result


@pytest.mark.asyncio
async def test_web_agent3_web_search_reports_attempt_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Malformed attempt-planner output should become visible weak coverage."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )
    fake_attempt_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(content="not json"))
    )
    monkeypatch.setattr(web_search_subagent, "_search_attempt_llm", fake_attempt_llm)
    monkeypatch.setattr(
        web_search_subagent,
        "parse_llm_json_output",
        MagicMock(side_effect=ValueError("malformed planner output")),
    )
    search_calls: list[dict[str, str]] = []

    async def fake_search(payload: dict[str, str]) -> str:
        search_calls.append(payload)
        return "Title: Broad source\nURL: https://example.test/broad"

    fake_search_tool = SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_search))
    monkeypatch.setattr(
        web_search_subagent.searxng_tools,
        "web_search",
        fake_search_tool,
    )
    dense_query = "Compare alpha, beta, and gamma across current docs."
    decision = web_module._RouterDecision(
        action="search",
        source="web_search",
        query=dense_query,
    )

    result = await web_search_subagent.execute(decision)

    assert search_calls == [{"query": dense_query}]
    assert "Search-attempt expansion produced no valid focused queries" in result


def test_web_agent3_web_search_detects_chinese_dense_query() -> None:
    """Dense-query detection should cover Chinese-first search wording."""
    web_search_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_search"
    )

    dense_queries = (
        "比较 Alpha、Beta 和 Gamma 在价格、支持状态以及版本差异上的情况",
        "比较Alpha和Beta的价格和支持状态",
        "Alpha和Beta性能对比",
        "Alpha与Beta价格对比",
    )
    for query in dense_queries:
        assert web_search_subagent._query_needs_attempt_expansion(query)


@pytest.mark.asyncio
async def test_web_agent3_web_read_receives_query_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """web_read should pass router query directly to the URL reader."""
    web_read_subagent = importlib.import_module(
        "kazusa_ai_chatbot.rag.web_agent3.subagent.web_read"
    )
    fake_read = SimpleNamespace(ainvoke=AsyncMock(return_value="page body"))
    monkeypatch.setattr(web_read_subagent.searxng_tools, "web_url_read", fake_read)
    decision = web_module._RouterDecision(
        action="read",
        source="web_read",
        query="https://example.test/page",
    )

    result = await web_module._execute_source_decision(decision)

    fake_read.ainvoke.assert_awaited_once_with({"url": "https://example.test/page"})
    assert result == "page body"


@pytest.mark.asyncio
async def test_web_agent3_executor_records_minimal_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should keep prompt-facing state minimal."""
    execute_decision = AsyncMock(return_value="page body")
    monkeypatch.setattr(agent_module, "_execute_source_decision", execute_decision)
    state = {
        "router_decision": {
            "action": "read",
            "source": "nhentai",
            "query": "652244",
        },
        "observations": [],
    }

    update = await agent_module._tool_call_executor(state)

    execute_decision.assert_awaited_once_with(web_module._RouterDecision(
        action="read",
        source="nhentai",
        query="652244",
    ))
    record = json.loads(update["messages"][0].content)
    assert record == {
        "action": "read",
        "source": "nhentai",
        "query": "652244",
        "result": "page body",
    }
    assert update["observations"] == [record]


@pytest.mark.asyncio
async def test_web_agent3_evaluator_continues_with_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evaluator feedback should feed the next router iteration."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content='{"should_stop": false, "feedback": "read the first URL"}',
        )),
    )
    monkeypatch.setattr(agent_module, "_evaluator_llm", fake_llm)
    state = {
        "task": "Find current docs.",
        "expected_response": "official status",
        "messages": [],
        "observations": [
            {
                "action": "search",
                "source": "web_search",
                "query": "current docs",
                "result": "search result",
            }
        ],
        "retry": 0,
        "prompt_timestamp": "2026-05-25 21:30 (Monday)",
    }

    update = await agent_module._tool_call_evaluator(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    system_prompt = messages[0].content
    payload = json.loads(messages[1].content)
    assert "`retry`" not in system_prompt
    assert payload["reference_time"] == "2026-05-25 21:30 (Monday)"
    assert payload["call_history"][0]["query"] == "current docs"
    assert "retry" not in payload
    assert update["should_stop"] is False
    assert update["evaluator_feedback"] == "read the first URL"
    assert update["retry"] == 1


@pytest.mark.asyncio
async def test_web_agent3_run_subgraph_returns_expected_keys() -> None:
    """_run_subgraph should map compiled graph state to the public result shape."""
    mock_result = {
        "final_status": "success",
        "final_reason": "found info",
        "final_response": "Here are the results",
        "final_is_empty_result": False,
        "knowledge_metadata": {},
    }

    with patch("kazusa_ai_chatbot.rag.web_agent3.agent.StateGraph") as state_graph:
        graph_builder = MagicMock()
        graph_builder.compile.return_value.ainvoke = AsyncMock(return_value=mock_result)
        state_graph.return_value = graph_builder

        result = await agent_module._run_subgraph(
            task="search something",
            context={
                "original_query": "用户想确认当前官方文档。",
                "current_slot": "official current docs",
                "channel_topic": "debug planning",
                "platform": "debug",
                "platform_channel_id": "raw-channel-123",
                "global_user_id": "raw-user-123",
                "platform_user_id": "raw-platform-user-123",
                "platform_bot_id": "raw-bot-123",
            },
            expected_response="relevant results",
            local_prompt_timestamp="2026-04-27 12:00",
        )

    sub_state = graph_builder.compile.return_value.ainvoke.await_args.args[0]
    assert sub_state["prompt_timestamp"] == "2026-04-27 12:00"
    assert sub_state["router_decision"] == {
        "action": "stop",
        "source": "web_read",
        "query": "",
    }
    assert sub_state["context"] == {
        "original_query": "用户想确认当前官方文档。",
        "current_slot": "official current docs",
        "channel_topic": "debug planning",
    }
    serialized_context = json.dumps(sub_state["context"], ensure_ascii=False)
    assert "raw-channel-123" not in serialized_context
    assert "raw-user-123" not in serialized_context
    assert "raw-platform-user-123" not in serialized_context
    assert "raw-bot-123" not in serialized_context
    assert result == {
        "status": "success",
        "reason": "found info",
        "response": "Here are the results",
        "is_empty_result": False,
        "knowledge_metadata": {},
    }


@pytest.mark.asyncio
async def test_web_agent3_run_preserves_base_helper_contract() -> None:
    """WebAgent3.run should expose the BaseRAG helper contract."""
    with patch(
        "kazusa_ai_chatbot.rag.web_agent3.agent._run_subgraph",
        new_callable=AsyncMock,
        return_value={
            "status": "success",
            "reason": "found info",
            "response": "evidence package",
            "is_empty_result": False,
            "knowledge_metadata": {},
        },
    ) as run_subgraph:
        turn_clock = build_turn_clock_from_storage_utc(
            "2026-04-27T00:00:00+00:00",
        )
        result = await WebAgent3().run(
            task="search current weather",
            context={"local_time_context": turn_clock["local_time_context"]},
        )

    run_subgraph.assert_awaited_once()
    assert result == {
        "resolved": True,
        "status": "success",
        "reason": "found info",
        "result": "evidence package",
        "attempts": 1,
        "knowledge_metadata": {},
        "cache": {
            "enabled": False,
            "hit": False,
            "cache_name": "",
            "reason": "agent_not_cacheable",
        },
    }


@pytest.mark.asyncio
async def test_web_agent3_run_preserves_partial_finalizer_status() -> None:
    """Partial finalizer output should not be promoted to resolved evidence."""

    with patch(
        "kazusa_ai_chatbot.rag.web_agent3.agent._run_subgraph",
        new_callable=AsyncMock,
        return_value={
            "status": "partial",
            "reason": "source is relevant but lacks current pricing",
            "response": "Found an old product page but no current price.",
            "is_empty_result": False,
            "knowledge_metadata": {"source_fit": "partial"},
        },
    ):
        result = await WebAgent3().run(
            task="Find current public pricing.",
            context={},
        )

    assert result["resolved"] is False
    assert result["status"] == "partial"
    assert result["reason"] == "source is relevant but lacks current pricing"
    assert result["result"] == "Found an old product page but no current price."
    assert result["knowledge_metadata"] == {"source_fit": "partial"}


@pytest.mark.asyncio
async def test_web_agent3_run_returns_empty_search_without_deterministic_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty search graph should not invent a deterministic read target."""

    run_subgraph = AsyncMock(return_value={
        "status": "not_found",
        "reason": "search found no source",
        "response": "No information retrieved.",
        "is_empty_result": True,
        "knowledge_metadata": {},
    })
    execute_source = AsyncMock()
    monkeypatch.setattr(agent_module, "_run_subgraph", run_subgraph)
    monkeypatch.setattr(agent_module, "_execute_source_decision", execute_source)

    result = await WebAgent3().run(
        task="Confirm the latest SampleTool GitHub release date.",
        context={},
    )

    execute_source.assert_not_awaited()
    assert result["resolved"] is False
    assert result["result"] == "No information retrieved."


@pytest.mark.asyncio
async def test_web_agent3_finalizer_comparison_helper_returns_public_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Comparison finalizer helper should preserve the web_agent3 result shape."""
    fake_finalizer = AsyncMock(return_value={
        "final_status": "success",
        "final_reason": "enough",
        "final_response": "证据：example",
        "final_is_empty_result": False,
    })
    monkeypatch.setattr(agent_module, "_tool_call_finalizer", fake_finalizer)
    tool_result = web_module._WebToolResult(
        resolved=True,
        operation="search",
        query="example query",
        url=None,
        title=None,
        description=None,
        content="",
        items=[
            web_module._WebSearchItem(
                title="Example",
                url="https://example.test",
                snippet="snippet",
                source="fixture",
            )
        ],
        delegation_reason=None,
        missing_context=[],
        error=None,
    )

    result = await web_module._finalize_web_agent3_result(
        task="Web-evidence: example",
        context={},
        local_prompt_timestamp="2026-05-26 12:00",
        tool_result=tool_result,
        evaluator_feedback="snippet-only evidence",
        evidence_limitations=["snippet_only"],
        max_status="partial",
    )

    assert result == {
        "status": "partial",
        "reason": "enough",
        "response": "证据：example",
        "is_empty_result": False,
        "knowledge_metadata": {
            "evidence_limitations": ["snippet_only"],
            "max_status": "partial",
        },
    }


def test_web_agent3_finalizer_prompt_covers_known_regression_rules() -> None:
    """Finalizer prompt should cover absent evidence, stale facts, and failures."""
    prompt = agent_module._WEB_AGENT3_FINALIZER_PROMPT

    assert "reference_time" in prompt
    assert "缺失" in prompt
    assert "未提及" in prompt
    assert "is_empty_result" in prompt
    assert "最新" in prompt
    assert "当前" in prompt
    assert "读取失败" in prompt
    assert "搜索摘要" in prompt
    assert "Search attempts" in prompt
    assert "每次尝试" in prompt
    assert "模型知识" in prompt
    assert "某个来源类别读取失败" in prompt
    assert "不得说没有冲突或信息一致" in prompt
    assert "不要把相邻产品、派生轨道、集成说明" in prompt


def test_web_agent3_evaluator_prompt_avoids_duplicate_empty_reads() -> None:
    """Evaluator prompt should avoid repeating empty reads of the same target."""
    prompt = agent_module._WEB_AGENT3_EVALUATOR_PROMPT

    assert "不要重复读取" in prompt
    assert "同一 URL" in prompt
    assert "正文为空" in prompt


@pytest.mark.asyncio
async def test_web_agent3_finalizer_empty_result_forces_not_found_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizer status should agree with an explicit empty-result decision."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content=(
                '{"response": "未找到相关信息", "score": 95, '
                '"reason": "content states the requested fact is absent", '
                '"is_empty_result": true}'
            ),
        )),
    )
    monkeypatch.setattr(agent_module, "_finalizer_llm", fake_llm)
    state = {
        "task": "Web-evidence: find operating hours",
        "expected_response": "operating hours",
        "messages": [
            ToolMessage(
                content='{"result": "page does not mention operating hours"}',
                tool_call_id="tool-1",
            ),
        ],
        "evaluator_feedback": "requested fact is absent",
        "prompt_timestamp": "2026-05-27 12:00 (Wednesday)",
    }

    update = await agent_module._tool_call_finalizer(state)

    assert update["final_status"] == "not_found"
    assert update["final_is_empty_result"] is True


@pytest.mark.asyncio
async def test_web_agent3_finalizer_missing_empty_flag_uses_score_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing empty-result flag with score 0 should not become resolved."""

    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content=(
                '{"response": "未找到任何有效证据", "score": 0, '
                '"reason": "all searches returned no results"}'
            ),
        )),
    )
    monkeypatch.setattr(agent_module, "_finalizer_llm", fake_llm)
    state = {
        "task": "Web-evidence: compare local-first note stacks",
        "expected_response": "source-backed comparison",
        "messages": [
            ToolMessage(
                content='{"result": "No results found"}',
                tool_call_id="tool-1",
            ),
        ],
        "evaluator_feedback": "no source evidence",
        "prompt_timestamp": "2026-05-27 12:00 (Wednesday)",
    }

    update = await agent_module._tool_call_finalizer(state)

    assert update["final_status"] == "not_found"
    assert update["final_is_empty_result"] is True


@pytest.mark.asyncio
async def test_web_agent3_finalizer_payload_uses_clean_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Finalizer should not receive evaluator message wrapper metadata."""
    fake_llm = SimpleNamespace(
        ainvoke=AsyncMock(return_value=AIMessage(
            content=(
                '{"response": "证据：example", "score": 90, '
                '"reason": "enough", "is_empty_result": false}'
            ),
        )),
    )
    monkeypatch.setattr(agent_module, "_finalizer_llm", fake_llm)
    feedback_payload = {
        "feedback": "read official page",
        "source": "evaluator",
        "evidence_limitations": ["snippet_only"],
    }
    state = {
        "task": "Web-evidence: example",
        "expected_response": "official evidence",
        "messages": [
            ToolMessage(
                content='{"result": "official page"}',
                tool_call_id="tool-1",
            ),
            HumanMessage(
                content=json.dumps(feedback_payload, ensure_ascii=False),
                name="evaluator",
            ),
        ],
        "evaluator_feedback": "read official page",
        "prompt_timestamp": "2026-05-27 12:00 (Wednesday)",
    }

    await agent_module._tool_call_finalizer(state)

    messages = fake_llm.ainvoke.await_args.args[0]
    finalizer_payload = json.loads(messages[1].content)
    assert finalizer_payload["reference_time"] == "2026-05-27 12:00 (Wednesday)"
    assert finalizer_payload["evaluator_feedback"] == "read official page"
    assert "source" not in messages[1].content
    assert "evidence_limitations" not in messages[1].content
