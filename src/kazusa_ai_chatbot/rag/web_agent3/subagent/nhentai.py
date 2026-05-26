"""nHentai source subagent backed by bounded metadata/search API calls."""

from __future__ import annotations

from http import HTTPStatus
import os
import re
from typing import Any

import httpx

from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

SOURCE = "nhentai"
DESCRIPTION = '''nHentai 图库元数据读取和图库搜索。
生成 query 时：
- search: 保留用户给出的图库搜索词、标签筛选、精确短语、排除词或数字编号。
- read: 保留原始数字编号、图库页面 URL 或用户给出的图库目标字符串。
'''

_NHENTAI_API_BASE_URL = "https://nhentai.net/api/v2"
_NHENTAI_PUBLIC_BASE_URL = "https://nhentai.net/g"
_NHENTAI_USER_AGENT = "kazusa_ai_chatbot/web_agent3 (local metadata lookup)"
_NHENTAI_TIMEOUT_SECONDS = 20.0
_NHENTAI_SEARCH_PAGE = 1
_NHENTAI_SEARCH_LIMIT = 5
_NHENTAI_TAG_GROUPS = (
    "language",
    "artist",
    "group",
    "parody",
    "character",
    "tag",
    "category",
)
_BARE_GALLERY_ID_RE = re.compile(r"^\s*(\d+)\s*$")
_GALLERY_URL_RE = re.compile(r"(?:^|/)g/(\d+)(?:/|$|\?)", re.IGNORECASE)
_API_GALLERY_RE = re.compile(
    r"/api/v2/galleries/(\d+)(?:/|$|\?)",
    re.IGNORECASE,
)


def _extract_gallery_id(raw_target: str) -> int | None:
    """Extract a gallery id from a bare number, gallery URL, or API URL."""
    stripped_target = raw_target.strip()
    gallery_match = _BARE_GALLERY_ID_RE.search(stripped_target)
    if gallery_match is None:
        gallery_match = _GALLERY_URL_RE.search(stripped_target)
    if gallery_match is None:
        gallery_match = _API_GALLERY_RE.search(stripped_target)

    if gallery_match is None:
        gallery_id = None
        return gallery_id

    gallery_id = int(gallery_match.group(1))
    return gallery_id


def _gallery_url(gallery_id: int) -> str:
    """Build the public gallery URL shown in compact observations."""
    url = f"{_NHENTAI_PUBLIC_BASE_URL}/{gallery_id}/"
    return url


def _headers_for_request() -> dict[str, str]:
    """Build per-request API headers without exposing credentials to callers."""
    headers = {"User-Agent": _NHENTAI_USER_AGENT}
    token = (os.getenv("NHENTAI_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Key {token}"

    return headers


async def _get_api_response(
    path: str,
    *,
    params: dict[str, Any] | None = None,
) -> httpx.Response:
    """Run one nHentai API GET request and return the raw response object."""
    url = f"{_NHENTAI_API_BASE_URL}{path}"
    headers = _headers_for_request()
    async with httpx.AsyncClient(
        headers=headers,
        timeout=_NHENTAI_TIMEOUT_SECONDS,
    ) as client:
        response = await client.get(url, params=params)

    return response


def _error_result(
    decision: _RouterDecision,
    *,
    gallery_id: int | None,
    status: str,
    message: str,
) -> dict[str, Any]:
    """Build a bounded error-like observation for the selected source."""
    url = None
    if gallery_id is not None:
        url = _gallery_url(gallery_id)

    result = {
        "status": status,
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "gallery_id": gallery_id,
        "url": url,
        "message": message,
    }
    return result


def _http_status_error_result(
    decision: _RouterDecision,
    *,
    gallery_id: int | None,
    status_code: int,
) -> dict[str, Any]:
    """Map API HTTP status codes to bounded source observations."""
    if status_code == HTTPStatus.NOT_FOUND:
        result = _error_result(
            decision,
            gallery_id=gallery_id,
            status="not_found",
            message="Gallery not found.",
        )
        return result

    if status_code == HTTPStatus.TOO_MANY_REQUESTS:
        result = _error_result(
            decision,
            gallery_id=gallery_id,
            status="error",
            message="nHentai API rate limited.",
        )
        return result

    result = _error_result(
        decision,
        gallery_id=gallery_id,
        status="error",
        message=f"nHentai API request failed with HTTP {status_code}.",
    )
    return result


def _json_payload_or_error(
    decision: _RouterDecision,
    *,
    gallery_id: int | None,
    response: httpx.Response,
) -> dict[str, Any] | Any:
    """Parse an API JSON response or return a bounded JSON error."""
    try:
        payload = response.json()
    except ValueError as exc:
        result = _error_result(
            decision,
            gallery_id=gallery_id,
            status="error",
            message=f"Invalid nHentai API JSON response: {exc}",
        )
        return result

    return payload


def _text_or_none(raw_value: object) -> str | None:
    """Normalize optional external text fields without inventing content."""
    if isinstance(raw_value, str):
        normalized_value = raw_value
    else:
        normalized_value = None

    return normalized_value


def _compact_title(title_payload: object) -> dict[str, str | None] | None:
    """Compact the API title object to the allowed title fields."""
    if not isinstance(title_payload, dict):
        compact_title = None
        return compact_title

    compact_title = {
        "english": _text_or_none(title_payload.get("english")),
        "japanese": _text_or_none(title_payload.get("japanese")),
        "pretty": _text_or_none(title_payload.get("pretty")),
    }
    return compact_title


def _compact_tags(tags_payload: object) -> dict[str, list[str]] | None:
    """Group gallery tag names by the stable nHentai tag type."""
    if not isinstance(tags_payload, list):
        compact_tags = None
        return compact_tags

    compact_tags: dict[str, list[str]] = {}
    for tag_group in _NHENTAI_TAG_GROUPS:
        compact_tags[tag_group] = []

    for raw_tag in tags_payload:
        if not isinstance(raw_tag, dict):
            continue

        raw_tag_type = raw_tag.get("type")
        raw_tag_name = raw_tag.get("name")
        if raw_tag_type not in compact_tags:
            continue
        if not isinstance(raw_tag_name, str):
            continue

        compact_tags[raw_tag_type].append(raw_tag_name)

    return compact_tags


def _compact_gallery_result(
    decision: _RouterDecision,
    *,
    expected_gallery_id: int,
    payload: object,
) -> dict[str, Any]:
    """Compact gallery metadata to title and grouped tag observations."""
    if not isinstance(payload, dict):
        result = _error_result(
            decision,
            gallery_id=expected_gallery_id,
            status="error",
            message="Unexpected nHentai API response shape.",
        )
        return result

    raw_gallery_id = payload.get("id")
    if not isinstance(raw_gallery_id, int):
        result = _error_result(
            decision,
            gallery_id=expected_gallery_id,
            status="error",
            message="Unexpected nHentai API response shape.",
        )
        return result

    compact_title = _compact_title(payload.get("title"))
    compact_tags = _compact_tags(payload.get("tags"))
    if compact_title is None or compact_tags is None:
        result = _error_result(
            decision,
            gallery_id=expected_gallery_id,
            status="error",
            message="Unexpected nHentai API response shape.",
        )
        return result

    result = {
        "status": "success",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "gallery_id": raw_gallery_id,
        "url": _gallery_url(raw_gallery_id),
        "title": compact_title,
        "tags": compact_tags,
        "message": "Gallery metadata loaded.",
    }
    return result


async def _read_gallery(
    decision: _RouterDecision,
    *,
    gallery_id: int,
) -> dict[str, Any]:
    """Read one gallery metadata record from the official API."""
    try:
        response = await _get_api_response(f"/galleries/{gallery_id}")
    except httpx.HTTPError as exc:
        result = _error_result(
            decision,
            gallery_id=gallery_id,
            status="error",
            message=f"nHentai API request failed: {exc}",
        )
        return result

    if response.status_code != HTTPStatus.OK:
        result = _http_status_error_result(
            decision,
            gallery_id=gallery_id,
            status_code=response.status_code,
        )
        return result

    payload_or_error = _json_payload_or_error(
        decision,
        gallery_id=gallery_id,
        response=response,
    )
    if isinstance(payload_or_error, dict):
        payload_status = payload_or_error.get("status")
        if payload_status == "error":
            return_value = payload_or_error
            return return_value

    result = _compact_gallery_result(
        decision,
        expected_gallery_id=gallery_id,
        payload=payload_or_error,
    )
    return result


def _compact_search_result(raw_gallery: object) -> dict[str, Any] | None:
    """Compact one search row to the candidate fields consumed by RAG."""
    if not isinstance(raw_gallery, dict):
        compact_result = None
        return compact_result

    raw_gallery_id = raw_gallery.get("id")
    raw_title = raw_gallery.get("english_title")
    raw_num_pages = raw_gallery.get("num_pages")
    raw_num_favorites = raw_gallery.get("num_favorites")
    if not isinstance(raw_gallery_id, int):
        compact_result = None
        return compact_result
    if not isinstance(raw_title, str):
        compact_result = None
        return compact_result
    if not isinstance(raw_num_pages, int):
        compact_result = None
        return compact_result
    if not isinstance(raw_num_favorites, int):
        compact_result = None
        return compact_result

    compact_result = {
        "id": raw_gallery_id,
        "url": _gallery_url(raw_gallery_id),
        "title": raw_title,
        "num_pages": raw_num_pages,
        "num_favorites": raw_num_favorites,
    }
    return compact_result


def _compact_search_results(payload: object) -> list[dict[str, Any]] | None:
    """Compact and cap search rows from the API search payload."""
    if not isinstance(payload, dict):
        compact_results = None
        return compact_results

    raw_results = payload.get("result")
    if not isinstance(raw_results, list):
        compact_results = None
        return compact_results

    compact_results: list[dict[str, Any]] = []
    for raw_gallery in raw_results:
        compact_result = _compact_search_result(raw_gallery)
        if compact_result is None:
            continue

        compact_results.append(compact_result)
        if len(compact_results) >= _NHENTAI_SEARCH_LIMIT:
            break

    return compact_results


async def _search_galleries(decision: _RouterDecision) -> dict[str, Any]:
    """Search galleries and return bounded candidate metadata."""
    params = {
        "query": decision.query,
        "sort": "date",
        "page": _NHENTAI_SEARCH_PAGE,
    }
    try:
        response = await _get_api_response("/search", params=params)
    except httpx.HTTPError as exc:
        result = _error_result(
            decision,
            gallery_id=None,
            status="error",
            message=f"nHentai API request failed: {exc}",
        )
        return result

    if response.status_code != HTTPStatus.OK:
        result = _http_status_error_result(
            decision,
            gallery_id=None,
            status_code=response.status_code,
        )
        return result

    payload_or_error = _json_payload_or_error(
        decision,
        gallery_id=None,
        response=response,
    )
    if isinstance(payload_or_error, dict):
        payload_status = payload_or_error.get("status")
        if payload_status == "error":
            return_value = payload_or_error
            return return_value

    compact_results = _compact_search_results(payload_or_error)
    if compact_results is None:
        result = _error_result(
            decision,
            gallery_id=None,
            status="error",
            message="Unexpected nHentai API response shape.",
        )
        return result

    result = {
        "status": "success",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "message": "Gallery search completed.",
        "results": compact_results,
    }
    return result


async def execute(decision: _RouterDecision) -> dict[str, Any]:
    """Execute nHentai read/search decisions with source-local parsing."""
    if decision.action == "stop":
        result = {
            "status": "stopped",
            "source": decision.source,
            "action": decision.action,
            "query": decision.query,
            "message": "Router stopped without another nHentai action.",
        }
        return result

    gallery_id = _extract_gallery_id(decision.query)
    if decision.action == "read":
        if gallery_id is None:
            result = _error_result(
                decision,
                gallery_id=None,
                status="error",
                message="No nHentai gallery id found in read target.",
            )
            return result

        result = await _read_gallery(decision, gallery_id=gallery_id)
        return result

    if gallery_id is not None:
        result = await _read_gallery(decision, gallery_id=gallery_id)
        return result

    result = await _search_galleries(decision)
    return result
