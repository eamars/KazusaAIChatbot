"""Deterministic tests for the web_agent3 Bilibili source subagent."""

from __future__ import annotations

import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.rag import web_agent3 as web_module
from kazusa_ai_chatbot.rag.web_agent3.subagent import bilibili as bilibili_subagent


class _FakeResponse:
    """Small JSON response fixture for subtitle body fetches."""

    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return_value = self._payload
        return return_value


def _install_fake_root(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Install a fake bilibili_api root module and return selected clients."""
    selected_clients: list[str] = []
    root = ModuleType("bilibili_api")

    def select_client(name: str) -> None:
        selected_clients.append(name)

    root.select_client = select_client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bilibili_api", root)
    return selected_clients


def _install_fake_video_module(
    monkeypatch: pytest.MonkeyPatch,
    *,
    subtitle_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Install a fake bilibili_api.video module."""
    calls: list[dict[str, Any]] = []
    video_module = ModuleType("bilibili_api.video")

    class FakeVideo:
        def __init__(self, *, bvid: str | None = None, aid: int | None = None) -> None:
            calls.append({"method": "__init__", "bvid": bvid, "aid": aid})

        async def get_info(self) -> dict[str, Any]:
            calls.append({"method": "get_info"})
            payload = {
                "title": "Vibe Coding Demo",
                "desc": "Video description with useful context.",
                "owner": {"name": "demo-up"},
                "stat": {
                    "view": 1000,
                    "danmaku": 3,
                    "reply": 5,
                    "favorite": 20,
                    "coin": 7,
                    "share": 2,
                    "like": 80,
                },
                "duration": 125,
                "pubdate": 1780368000,
            }
            return payload

        async def get_pages(self) -> list[dict[str, Any]]:
            calls.append({"method": "get_pages"})
            return [{"cid": 222, "page": 1, "part": "main"}]

        async def get_subtitle(self, cid: int) -> dict[str, Any]:
            calls.append({"method": "get_subtitle", "cid": cid})
            return {
                "subtitles": [
                    {
                        "subtitle_url": "https://subtitle.test/body.json",
                        "lan": "zh-CN",
                    }
                ],
            }

    video_module.Video = FakeVideo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bilibili_api.video", video_module)

    class FakeSubtitleClient:
        def __init__(self, *, timeout: float) -> None:
            calls.append({"method": "subtitle_client", "timeout": timeout})

        async def __aenter__(self) -> "FakeSubtitleClient":
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            calls.append({"method": "subtitle_get", "url": url})
            payload = subtitle_payload or {
                "body": [
                    {"content": "first subtitle line"},
                    {"content": "second subtitle line"},
                ]
            }
            return _FakeResponse(payload)

    monkeypatch.setattr(bilibili_subagent.httpx, "AsyncClient", FakeSubtitleClient)
    return calls


def _install_fake_search_module(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Install a fake bilibili_api.search module."""
    calls: list[dict[str, Any]] = []
    search_module = ModuleType("bilibili_api.search")

    async def search(keyword: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({"method": "search", "keyword": keyword, "kwargs": kwargs})
        return {
            "result": [
                {
                    "result_type": "video",
                    "data": [
                        {
                            "title": "General Result",
                            "arcurl": "https://www.bilibili.com/video/BV1111111111/",
                            "type": "video",
                            "author": "general-up",
                            "description": "general summary",
                            "play": 200,
                        }
                    ],
                }
            ]
        }

    async def search_by_type(keyword: str, **kwargs: Any) -> dict[str, Any]:
        calls.append({
            "method": "search_by_type",
            "keyword": keyword,
            "kwargs": kwargs,
        })
        return {
            "result": [
                {
                    "title": "Popular Video",
                    "arcurl": "https://www.bilibili.com/video/BV2222222222/",
                    "bvid": "BV2222222222",
                    "author": "popular-up",
                    "description": "popular summary",
                    "play": 999,
                    "favorites": 88,
                }
            ]
        }

    search_module.search = search  # type: ignore[attr-defined]
    search_module.search_by_type = search_by_type  # type: ignore[attr-defined]
    search_module.SearchObjectType = SimpleNamespace(  # type: ignore[attr-defined]
        VIDEO="VIDEO",
        ARTICLE="ARTICLE",
        LIVE="LIVE",
        BANGUMI="BANGUMI",
        USER="USER",
    )
    search_module.OrderVideo = SimpleNamespace(  # type: ignore[attr-defined]
        TOTALRANK="TOTALRANK",
        CLICK="CLICK",
    )
    monkeypatch.setitem(sys.modules, "bilibili_api.search", search_module)
    return calls


def _install_fake_article_module(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Install a fake bilibili_api.article module."""
    calls: list[dict[str, Any]] = []
    article_module = ModuleType("bilibili_api.article")

    class FakeArticle:
        def __init__(self, **kwargs: Any) -> None:
            calls.append({"method": "__init__", "kwargs": kwargs})

        async def get_info(self) -> dict[str, Any]:
            calls.append({"method": "get_info"})
            return {
                "title": '专栏标题',
                "summary": '专栏摘要',
                "author": {"name": '作者'},
                "stats": {"view": 12, "like": 3},
            }

    article_module.Article = FakeArticle  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bilibili_api.article", article_module)
    return calls


def _assert_no_forbidden_material(result: dict[str, Any]) -> None:
    """Assert compact observations omit account, raw payload, and binary details."""
    rendered = json.dumps(result, ensure_ascii=False)
    forbidden_terms = [
        "credential",
        "cookie",
        "download",
        "favorites",
        "comments",
        "subtitle_url",
        "body.json",
    ]

    for forbidden_term in forbidden_terms:
        assert forbidden_term not in rendered


def test_bilibili_is_disabled_without_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bilibili availability should follow optional SDK importability."""
    monkeypatch.setattr(
        bilibili_subagent.importlib.util,
        "find_spec",
        lambda name: None if name == "bilibili_api" else object(),
    )

    assert bilibili_subagent.is_enabled() is False


@pytest.mark.asyncio
async def test_bilibili_video_read_selects_httpx_client_and_compacts_subtitles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Video reads should use the SDK lazily and return bounded observations."""
    selected_clients = _install_fake_root(monkeypatch)
    calls = _install_fake_video_module(monkeypatch)
    decision = web_module._RouterDecision(
        action="read",
        source="bilibili",
        query="https://www.bilibili.com/video/BV1CqV266EJY/",
    )

    result = await bilibili_subagent.execute(decision)

    assert selected_clients == ["httpx"]
    assert calls == [
        {"method": "__init__", "bvid": "BV1CqV266EJY", "aid": None},
        {"method": "get_info"},
        {"method": "get_pages"},
        {"method": "get_subtitle", "cid": 222},
        {"method": "subtitle_client", "timeout": 20.0},
        {"method": "subtitle_get", "url": "https://subtitle.test/body.json"},
    ]
    assert result["status"] == "success"
    assert result["source"] == "bilibili"
    assert result["action"] == "read"
    assert result["content_type"] == "video"
    assert result["content_scope"] == "video"
    assert result["public_id"] == "BV1CqV266EJY"
    assert result["title"] == "Vibe Coding Demo"
    assert result["creator"] == "demo-up"
    assert result["summary"] == "Video description with useful context."
    assert result["stats_summary"] == {
        "view": 1000,
        "danmaku": 3,
        "reply": 5,
        "favorite": 20,
        "coin": 7,
        "share": 2,
        "like": 80,
    }
    assert result["duration_seconds"] == 125
    assert result["pages"] == [{"cid": 222, "page": 1, "title": "main"}]
    assert result["subtitle_excerpt"] == (
        "first subtitle line\nsecond subtitle line"
    )
    assert result["content_basis"] == ["metadata", "pages", "subtitle"]
    _assert_no_forbidden_material(result)


@pytest.mark.asyncio
async def test_bilibili_read_supports_non_video_url_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read dispatch should support non-video public URL families."""
    _install_fake_root(monkeypatch)
    calls = _install_fake_article_module(monkeypatch)
    decision = web_module._RouterDecision(
        action="read",
        source="bilibili",
        query="https://www.bilibili.com/read/cv12345",
    )

    result = await bilibili_subagent.execute(decision)

    assert calls == [
        {"method": "__init__", "kwargs": {"cvid": 12345}},
        {"method": "get_info"},
    ]
    assert result["status"] == "success"
    assert result["content_type"] == "article"
    assert result["content_scope"] == "article"
    assert result["public_id"] == "cv12345"
    assert result["title"] == '专栏标题'
    assert result["creator"] == '作者'
    assert result["summary"] == '专栏摘要'
    assert result["stats_summary"] == {"view": 12, "like": 3}
    assert result["content_basis"] == ["metadata"]
    _assert_no_forbidden_material(result)


@pytest.mark.asyncio
async def test_bilibili_read_returns_unsupported_for_unknown_public_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unknown Bilibili URL families should stop inside the source."""
    _install_fake_root(monkeypatch)
    decision = web_module._RouterDecision(
        action="read",
        source="bilibili",
        query="https://www.bilibili.com/unknown/123",
    )

    result = await bilibili_subagent.execute(decision)

    assert result == {
        "status": "unsupported",
        "source": "bilibili",
        "action": "read",
        "query": "https://www.bilibili.com/unknown/123",
        "content_type": "unknown",
        "content_scope": "unknown",
        "public_id": None,
        "url": "https://www.bilibili.com/unknown/123",
        "message": "Unsupported or unrecognized Bilibili public target.",
    }


@pytest.mark.asyncio
async def test_bilibili_search_uses_general_search_for_semantic_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unscoped semantic searches should use provider general search."""
    _install_fake_root(monkeypatch)
    calls = _install_fake_search_module(monkeypatch)
    decision = web_module._RouterDecision(
        action="search",
        source="bilibili",
        query='vibe coding',
    )

    result = await bilibili_subagent.execute(decision)

    assert calls == [{
        "method": "search",
        "keyword": "vibe coding",
        "kwargs": {"page": 1},
    }]
    assert result["status"] == "success"
    assert result["content_scope"] == "general"
    assert result["popularity_basis"] == "provider_default"
    assert result["results"] == [{
        "title": "General Result",
        "url": "https://www.bilibili.com/video/BV1111111111/",
        "content_type": "video",
        "public_id": "BV1111111111",
        "creator": "general-up",
        "summary": "general summary",
        "stats_summary": {"view": 200},
    }]
    _assert_no_forbidden_material(result)


@pytest.mark.asyncio
async def test_bilibili_search_uses_video_click_order_for_hot_video_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot video searches should map to typed video search by click order."""
    _install_fake_root(monkeypatch)
    calls = _install_fake_search_module(monkeypatch)
    decision = web_module._RouterDecision(
        action="search",
        source="bilibili",
        query='帮我在bilibili上搜索关于vibe coding相关视频并且推荐给我一个最热门的视频',
    )

    result = await bilibili_subagent.execute(decision)

    assert calls == [{
        "method": "search_by_type",
        "keyword": (
            '帮我在bilibili上搜索关于vibe coding相关视频并且推荐给我一个最热门的视频'
        ),
        "kwargs": {
            "search_type": "VIDEO",
            "order_type": "CLICK",
            "page": 1,
        },
    }]
    assert result["status"] == "success"
    assert result["content_scope"] == "video"
    assert result["popularity_basis"] == "most_clicked"
    assert result["results"][0]["title"] == "Popular Video"
    assert result["results"][0]["public_id"] == "BV2222222222"
    assert result["results"][0]["stats_summary"] == {
        "view": 999,
        "favorite": 88,
    }
    _assert_no_forbidden_material(result)


@pytest.mark.asyncio
async def test_bilibili_search_infers_video_from_hot_order_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hot-order intent should use video search when scope was omitted."""
    _install_fake_root(monkeypatch)
    calls = _install_fake_search_module(monkeypatch)
    decision = web_module._RouterDecision(
        action="search",
        source="bilibili",
        query='vibe coding 最热',
    )

    result = await bilibili_subagent.execute(decision)

    assert calls == [{
        "method": "search_by_type",
        "keyword": 'vibe coding 最热',
        "kwargs": {
            "search_type": "VIDEO",
            "order_type": "CLICK",
            "page": 1,
        },
    }]
    assert result["status"] == "success"
    assert result["content_scope"] == "video"
    assert result["popularity_basis"] == "most_clicked"
    _assert_no_forbidden_material(result)
