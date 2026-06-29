"""Deterministic tests for the web_agent3 nHentai source subagent."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from kazusa_ai_chatbot.rag import web_agent3 as web_module
from kazusa_ai_chatbot.rag.web_agent3.subagent import nhentai as nhentai_subagent
from kazusa_ai_chatbot import config


class _FakeResponse:
    """Small response object with the subset used by the nHentai subagent."""

    def __init__(
        self,
        *,
        status_code: int,
        payload: Any = None,
        json_error: ValueError | None = None,
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self._json_error = json_error

    def json(self) -> Any:
        """Return the fake JSON payload or raise the configured JSON error."""
        if self._json_error is not None:
            raise self._json_error

        return_value = self._payload
        return return_value


def _gallery_payload() -> dict[str, Any]:
    """Build a gallery detail payload that includes fields which must be dropped."""
    payload = {
        "id": 652244,
        "media_id": "123456",
        "title": {
            "english": "Sample English Title",
            "japanese": "サンプル日本語タイトル",
            "pretty": "Sample Pretty Title",
        },
        "cover": {
            "path": "https://i.nhentai.net/galleries/123456/cover.jpg",
            "width": 350,
            "height": 500,
        },
        "thumbnail": {
            "path": "https://t.nhentai.net/galleries/123456/thumb.jpg",
            "width": 250,
            "height": 360,
        },
        "upload_date": 1779804800,
        "tags": [
            {"id": 1, "type": "language", "name": "english", "slug": "english"},
            {"id": 2, "type": "artist", "name": "artist-a", "slug": "artist-a"},
            {"id": 3, "type": "group", "name": "group-a", "slug": "group-a"},
            {"id": 4, "type": "parody", "name": "original", "slug": "original"},
            {"id": 5, "type": "character", "name": "char-a", "slug": "char-a"},
            {"id": 6, "type": "tag", "name": "slice of life", "slug": "slice"},
            {"id": 7, "type": "category", "name": "doujinshi", "slug": "doujinshi"},
        ],
        "num_pages": 25,
        "num_favorites": 1000,
        "pages": [
            {
                "number": 1,
                "path": "https://i.nhentai.net/galleries/123456/1.jpg",
                "thumbnail": "https://t.nhentai.net/galleries/123456/1t.jpg",
            }
        ],
        "comments": [{"body": "must not leak"}],
        "is_favorited": True,
    }
    return payload


def _search_payload(count: int = 6) -> dict[str, Any]:
    """Build a gallery search payload with more rows than the output cap."""
    rows: list[dict[str, Any]] = []
    for index in range(count):
        gallery_id = 652240 + index
        rows.append({
            "id": gallery_id,
            "media_id": str(gallery_id),
            "english_title": f"Search Result {index}",
            "japanese_title": None,
            "thumbnail": "https://t.nhentai.net/galleries/123456/thumb.jpg",
            "thumbnail_width": 250,
            "thumbnail_height": 360,
            "num_pages": 20 + index,
            "num_favorites": 100 + index,
            "tag_ids": [1, 2, 3],
            "blacklisted": False,
        })

    payload = {
        "result": rows,
        "num_pages": 3,
        "per_page": 25,
        "total": count,
    }
    return payload


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    response: _FakeResponse | None = None,
    error: httpx.HTTPError | None = None,
) -> list[dict[str, Any]]:
    """Install a fake httpx client and return its captured request calls."""
    calls: list[dict[str, Any]] = []

    class _FakeAsyncClient:
        def __init__(self, *, headers: dict[str, str], timeout: float) -> None:
            self.headers = headers
            self.timeout = timeout

        async def __aenter__(self) -> "_FakeAsyncClient":
            return self

        async def __aexit__(
            self,
            exc_type: object,
            exc: object,
            traceback: object,
        ) -> None:
            return None

        async def get(
            self,
            url: str,
            *,
            params: dict[str, Any] | None = None,
        ) -> _FakeResponse:
            calls.append({
                "url": url,
                "params": params,
                "headers": self.headers,
                "timeout": self.timeout,
            })
            if error is not None:
                raise error

            assert response is not None
            return_value = response
            return return_value

    monkeypatch.setattr(nhentai_subagent.httpx, "AsyncClient", _FakeAsyncClient)
    return calls


def _assert_no_forbidden_material(result: dict[str, Any]) -> None:
    """Assert compact observations do not leak media, account, or auth data."""
    rendered = json.dumps(result, ensure_ascii=False)
    forbidden_terms = [
        "secret-token",
        "Authorization",
        "headers",
        "thumbnail",
        "cover",
        "download",
        "comments",
        "is_favorited",
        '"pages"',
        "i.nhentai.net",
        "t.nhentai.net",
    ]

    for forbidden_term in forbidden_terms:
        assert forbidden_term not in rendered


def test_nhentai_extracts_gallery_ids_from_supported_targets() -> None:
    """Gallery id extraction should stay inside the nHentai source subagent."""
    cases = [
        ("652244", 652244),
        (" https://nhentai.net/g/652244/ ", 652244),
        ("https://nhentai.net/api/v2/galleries/652244", 652244),
        ("/api/v2/galleries/652244?include=related", 652244),
        ("artist:demo language:english", None),
        ("", None),
    ]

    for raw_target, expected_id in cases:
        gallery_id = nhentai_subagent._extract_gallery_id(raw_target)

        assert gallery_id == expected_id


def test_nhentai_is_disabled_without_token_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """nHentai source availability should follow imported config state."""
    monkeypatch.setattr(nhentai_subagent, "NHENTAI_SOURCE_ENABLED", False)

    assert nhentai_subagent.is_enabled() is False


def test_nhentai_uses_token_from_config_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Request headers should use the token imported from config.py."""
    monkeypatch.setattr(nhentai_subagent, "NHENTAI_TOKEN", "secret-token")

    headers = nhentai_subagent._headers_for_request()

    assert headers == {
        "User-Agent": nhentai_subagent._NHENTAI_USER_AGENT,
        "Authorization": "Key secret-token",
    }


def test_nhentai_provider_constants_remain_source_local() -> None:
    """Stable nHentai provider constants should stay out of config.py."""
    assert nhentai_subagent._NHENTAI_API_BASE_URL == "https://nhentai.net/api/v2"
    assert nhentai_subagent._NHENTAI_PUBLIC_BASE_URL == "https://nhentai.net/g"
    assert nhentai_subagent._NHENTAI_USER_AGENT
    assert not hasattr(config, "NHENTAI_API_BASE_URL")
    assert not hasattr(config, "NHENTAI_PUBLIC_BASE_URL")
    assert not hasattr(config, "NHENTAI_USER_AGENT")


@pytest.mark.asyncio
async def test_nhentai_read_returns_title_and_grouped_tags_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Read should call the gallery API and compact to title plus tags."""
    monkeypatch.setattr(nhentai_subagent, "NHENTAI_TOKEN", "secret-token")
    calls = _install_fake_client(
        monkeypatch,
        response=_FakeResponse(status_code=200, payload=_gallery_payload()),
    )
    decision = web_module._RouterDecision(
        action="read",
        source="nhentai",
        query="https://nhentai.net/g/652244/",
    )

    result = await nhentai_subagent.execute(decision)

    assert calls == [{
        "url": "https://nhentai.net/api/v2/galleries/652244",
        "params": None,
        "headers": {
            "User-Agent": nhentai_subagent._NHENTAI_USER_AGENT,
            "Authorization": "Key secret-token",
        },
        "timeout": nhentai_subagent._NHENTAI_TIMEOUT_SECONDS,
    }]
    assert result == {
        "status": "success",
        "source": "nhentai",
        "action": "read",
        "query": "https://nhentai.net/g/652244/",
        "gallery_id": 652244,
        "url": "https://nhentai.net/g/652244/",
        "title": {
            "english": "Sample English Title",
            "japanese": "サンプル日本語タイトル",
            "pretty": "Sample Pretty Title",
        },
        "tags": {
            "language": ["english"],
            "artist": ["artist-a"],
            "group": ["group-a"],
            "parody": ["original"],
            "character": ["char-a"],
            "tag": ["slice of life"],
            "category": ["doujinshi"],
        },
        "message": "Gallery metadata loaded.",
    }
    _assert_no_forbidden_material(result)


@pytest.mark.asyncio
async def test_nhentai_search_returns_bounded_gallery_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Search should call the gallery search API and cap compact candidates."""
    monkeypatch.setattr(nhentai_subagent, "NHENTAI_TOKEN", "")
    calls = _install_fake_client(
        monkeypatch,
        response=_FakeResponse(status_code=200, payload=_search_payload()),
    )
    decision = web_module._RouterDecision(
        action="search",
        source="nhentai",
        query='artist:demo language:english -"excluded tag"',
    )

    result = await nhentai_subagent.execute(decision)

    assert calls == [{
        "url": "https://nhentai.net/api/v2/search",
        "params": {
            "query": 'artist:demo language:english -"excluded tag"',
            "sort": "date",
            "page": 1,
        },
        "headers": {
            "User-Agent": nhentai_subagent._NHENTAI_USER_AGENT,
        },
        "timeout": nhentai_subagent._NHENTAI_TIMEOUT_SECONDS,
    }]
    assert result["status"] == "success"
    assert result["source"] == "nhentai"
    assert result["action"] == "search"
    assert result["query"] == 'artist:demo language:english -"excluded tag"'
    assert result["message"] == "Gallery search completed."
    assert result["results"] == [
        {
            "id": 652240,
            "url": "https://nhentai.net/g/652240/",
            "title": "Search Result 0",
            "num_pages": 20,
            "num_favorites": 100,
        },
        {
            "id": 652241,
            "url": "https://nhentai.net/g/652241/",
            "title": "Search Result 1",
            "num_pages": 21,
            "num_favorites": 101,
        },
        {
            "id": 652242,
            "url": "https://nhentai.net/g/652242/",
            "title": "Search Result 2",
            "num_pages": 22,
            "num_favorites": 102,
        },
        {
            "id": 652243,
            "url": "https://nhentai.net/g/652243/",
            "title": "Search Result 3",
            "num_pages": 23,
            "num_favorites": 103,
        },
        {
            "id": 652244,
            "url": "https://nhentai.net/g/652244/",
            "title": "Search Result 4",
            "num_pages": 24,
            "num_favorites": 104,
        },
    ]
    _assert_no_forbidden_material(result)


@pytest.mark.asyncio
async def test_nhentai_numeric_search_uses_gallery_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A numeric search target should be resolved inside the nHentai subagent."""
    calls = _install_fake_client(
        monkeypatch,
        response=_FakeResponse(status_code=200, payload=_gallery_payload()),
    )
    decision = web_module._RouterDecision(
        action="search",
        source="nhentai",
        query="652244",
    )

    result = await nhentai_subagent.execute(decision)

    assert calls[0]["url"] == "https://nhentai.net/api/v2/galleries/652244"
    assert calls[0]["params"] is None
    assert result["status"] == "success"
    assert result["source"] == "nhentai"
    assert result["action"] == "search"
    assert result["gallery_id"] == 652244
    assert result["title"]["english"] == "Sample English Title"
    _assert_no_forbidden_material(result)


@pytest.mark.asyncio
async def test_nhentai_missing_gallery_id_returns_bounded_error() -> None:
    """Read without a gallery id should fail without using another source."""
    decision = web_module._RouterDecision(
        action="read",
        source="nhentai",
        query="artist:demo language:english",
    )

    result = await nhentai_subagent.execute(decision)

    assert result == {
        "status": "error",
        "source": "nhentai",
        "action": "read",
        "query": "artist:demo language:english",
        "gallery_id": None,
        "url": None,
        "message": "No nHentai gallery id found in read target.",
    }


@pytest.mark.asyncio
async def test_nhentai_api_errors_are_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """External API failures should return bounded observations."""
    cases = [
        (
            _FakeResponse(status_code=404, payload={"error": "not found"}),
            None,
            "not_found",
            "Gallery not found.",
        ),
        (
            _FakeResponse(status_code=429, payload={"error": "rate limit"}),
            None,
            "error",
            "nHentai API rate limited.",
        ),
        (
            None,
            httpx.TimeoutException("request timed out"),
            "error",
            "nHentai API request failed: request timed out",
        ),
        (
            _FakeResponse(
                status_code=200,
                json_error=ValueError("invalid json"),
            ),
            None,
            "error",
            "Invalid nHentai API JSON response: invalid json",
        ),
        (
            _FakeResponse(status_code=200, payload={"id": "not-an-int"}),
            None,
            "error",
            "Unexpected nHentai API response shape.",
        ),
    ]

    for response, error, expected_status, expected_message in cases:
        _install_fake_client(monkeypatch, response=response, error=error)
        decision = web_module._RouterDecision(
            action="read",
            source="nhentai",
            query="652244",
        )

        result = await nhentai_subagent.execute(decision)

        assert result["status"] == expected_status
        assert result["source"] == "nhentai"
        assert result["action"] == "read"
        assert result["query"] == "652244"
        assert result["gallery_id"] == 652244
        assert result["url"] == "https://nhentai.net/g/652244/"
        assert result["message"] == expected_message
        _assert_no_forbidden_material(result)
