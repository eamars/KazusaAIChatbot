"""Bilibili source subagent backed by optional public SDK reads/search."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import importlib.util
import inspect
import re
from types import ModuleType
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from kazusa_ai_chatbot.rag.web_agent3.contracts import _RouterDecision

SOURCE = "bilibili"
SUPPORTED_ACTIONS = ("read", "search")
DESCRIPTION = '''Bilibili 公共内容读取和站内语义搜索。
生成 query 时：
- search: 保留用户语义主题、内容范围词和排序意图词。
  用户指定视频、专栏、番剧、直播、UP 主、动态、音频或话题时，query 必须保留对应范围词。
  用户要求热门、最热、最多播放或最多点击时，query 必须保留对应排序意图。
- read: 保留原始 Bilibili URL、BV 号或 av 号；链接目标由 Bilibili source 内部解析为对应公共内容类型。
'''

_BILIBILI_PUBLIC_BASE_URL = "https://www.bilibili.com"
_SEARCH_PAGE = 1
_SEARCH_RESULT_LIMIT = 5
_SUBTITLE_TIMEOUT_SECONDS = 20.0
_SUBTITLE_CHAR_LIMIT = 1200
_TEXT_CHAR_LIMIT = 1200
_PAGE_LIMIT = 5
_BVID_RE = re.compile(r"\b(BV[0-9A-Za-z]{10})\b")
_AID_RE = re.compile(r"\b(?:av|AV)(\d+)\b")
_ARTICLE_ID_RE = re.compile(r"^cv(\d+)$", re.IGNORECASE)
_AUDIO_ID_RE = re.compile(r"^au(\d+)$", re.IGNORECASE)
_PREFIXED_ID_RE = re.compile(r"^([A-Za-z]+)(\d+)$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_READ_CONTENT_TYPES = (
    "video",
    "article",
    "bangumi",
    "live",
    "user",
    "dynamic",
    "audio",
    "topic",
)
_STAT_FIELDS = (
    ("view", ("view", "views", "play", "click")),
    ("danmaku", ("danmaku", "dm")),
    ("reply", ("reply", "replies", "comment", "comments")),
    ("favorite", ("favorite", "favorites", "fav")),
    ("coin", ("coin", "coins")),
    ("share", ("share", "shares")),
    ("like", ("like", "likes")),
)
_SEARCH_SCOPE_ENUM_NAMES = {
    "video": "VIDEO",
    "article": "ARTICLE",
    "live": "LIVE",
    "bangumi": "BANGUMI",
    "user": "USER",
}
_POPULAR_MARKERS = ('热门', '最热', '最多播放', '最多点击', 'hot', 'popular')


@dataclass(frozen=True)
class _ReadTarget:
    """Parsed Bilibili public read target."""

    content_type: str
    public_id: str | None
    numeric_id: int | None
    url: str | None


@dataclass(frozen=True)
class _SdkReadProtocol:
    """Best-effort read protocol for one Bilibili SDK content family."""

    module_name: str
    class_names: tuple[str, ...]
    kwargs_options: tuple[dict[str, Any], ...]
    method_names: tuple[str, ...]


def is_enabled() -> bool:
    """Return whether the optional bilibili-api-python dependency is installed."""
    try:
        sdk_spec = importlib.util.find_spec("bilibili_api")
    except (ImportError, ValueError):
        return_value = False
        return return_value

    return_value = sdk_spec is not None
    return return_value


def _load_sdk_root() -> ModuleType:
    """Import the optional SDK root and select this project's HTTP client."""
    sdk_root = importlib.import_module("bilibili_api")
    select_client = getattr(sdk_root, "select_client", None)
    if callable(select_client):
        select_client("httpx")

    return sdk_root


def _bilibili_api_error_types() -> tuple[type[BaseException], ...]:
    """Return the SDK's public API exception base when available."""
    try:
        exceptions_module = importlib.import_module("bilibili_api.exceptions")
    except ImportError:
        return_value: tuple[type[BaseException], ...] = ()
        return return_value

    api_exception = getattr(exceptions_module, "ApiException", None)
    if isinstance(api_exception, type) and issubclass(api_exception, BaseException):
        return_value = (api_exception,)
        return return_value

    return_value = ()
    return return_value


def _external_bilibili_error_types() -> tuple[type[BaseException], ...]:
    """Return bounded exception types for SDK and HTTP boundary calls."""
    error_types = (
        ImportError,
        AttributeError,
        TypeError,
        ValueError,
        RuntimeError,
        httpx.HTTPError,
    )
    return_value = error_types + _bilibili_api_error_types()
    return return_value


def _text_or_none(raw_value: object) -> str | None:
    """Normalize optional external text fields without inventing content."""
    if not isinstance(raw_value, str):
        return_value = None
        return return_value

    stripped_value = _HTML_TAG_RE.sub("", raw_value).strip()
    if not stripped_value:
        return_value = None
        return return_value

    return_value = stripped_value[:_TEXT_CHAR_LIMIT]
    return return_value


def _first_text(payload: dict[str, Any], field_names: tuple[str, ...]) -> str | None:
    """Return the first usable text field from an external payload."""
    for field_name in field_names:
        normalized_value = _text_or_none(payload.get(field_name))
        if normalized_value is not None:
            return normalized_value

    return_value = None
    return return_value


def _int_or_none(raw_value: object) -> int | None:
    """Normalize integer-like public count fields."""
    if isinstance(raw_value, bool):
        return_value = None
        return return_value

    if isinstance(raw_value, int):
        return_value = raw_value
        return return_value

    if isinstance(raw_value, str) and raw_value.isdigit():
        return_value = int(raw_value)
        return return_value

    return_value = None
    return return_value


def _compact_stats(payload: object) -> dict[str, int] | None:
    """Compact public stat counters to a stable prompt-safe summary."""
    if not isinstance(payload, dict):
        return_value = None
        return return_value

    stats: dict[str, int] = {}
    for output_name, input_names in _STAT_FIELDS:
        for input_name in input_names:
            raw_value = payload.get(input_name)
            normalized_value = _int_or_none(raw_value)
            if normalized_value is None:
                continue

            stats[output_name] = normalized_value
            break

    if not stats:
        return_value = None
        return return_value

    return stats


def _stats_payload(payload: dict[str, Any]) -> object:
    """Return the nested public stats payload when present."""
    for field_name in ("stat", "stats"):
        raw_stats = payload.get(field_name)
        if isinstance(raw_stats, dict):
            return raw_stats

    return payload


def _creator_from_payload(payload: dict[str, Any]) -> str | None:
    """Extract a compact public creator name from known response shapes."""
    for nested_field in ("owner", "author", "upper", "up"):
        raw_nested = payload.get(nested_field)
        if not isinstance(raw_nested, dict):
            continue

        creator = _first_text(raw_nested, ("name", "uname", "mid"))
        if creator is not None:
            return creator

    creator = _first_text(payload, ("author", "uname", "owner_name"))
    return creator


def _public_url_for_target(target: _ReadTarget) -> str | None:
    """Build the stable public URL for shorthand reads."""
    if target.url is not None:
        return_value = target.url
        return return_value

    if target.content_type == "video" and target.public_id is not None:
        return_value = f"{_BILIBILI_PUBLIC_BASE_URL}/video/{target.public_id}/"
        return return_value

    return_value = None
    return return_value


def _parse_prefixed_numeric_id(raw_value: str) -> tuple[str, int] | None:
    """Parse an id with a short alpha prefix such as cv, ep, ss, or au."""
    prefixed_match = _PREFIXED_ID_RE.match(raw_value)
    if prefixed_match is None:
        return_value = None
        return return_value

    parsed_value = (prefixed_match.group(1).lower(), int(prefixed_match.group(2)))
    return parsed_value


def _read_target_from_video_id(raw_query: str) -> _ReadTarget | None:
    """Extract a BV or av video id from a raw target."""
    bvid_match = _BVID_RE.search(raw_query)
    if bvid_match is not None:
        target = _ReadTarget(
            content_type="video",
            public_id=bvid_match.group(1),
            numeric_id=None,
            url=None,
        )
        return target

    aid_match = _AID_RE.search(raw_query)
    if aid_match is not None:
        aid = int(aid_match.group(1))
        target = _ReadTarget(
            content_type="video",
            public_id=f"av{aid}",
            numeric_id=aid,
            url=None,
        )
        return target

    target = None
    return target


def _parse_bilibili_url(raw_query: str) -> _ReadTarget | None:
    """Parse supported public Bilibili URL families."""
    parsed_url = urlparse(raw_query)
    host = parsed_url.netloc.lower()
    path_parts = [part for part in parsed_url.path.split("/") if part]
    if "bilibili.com" not in host and host != "b23.tv":
        target = None
        return target

    video_target = _read_target_from_video_id(raw_query)
    if video_target is not None:
        target = _ReadTarget(
            content_type=video_target.content_type,
            public_id=video_target.public_id,
            numeric_id=video_target.numeric_id,
            url=raw_query.strip(),
        )
        return target

    if host == "live.bilibili.com" and path_parts:
        room_id = _int_or_none(path_parts[0])
        if room_id is not None:
            target = _ReadTarget("live", str(room_id), room_id, raw_query.strip())
            return target

    if host == "space.bilibili.com" and path_parts:
        uid = _int_or_none(path_parts[0])
        if uid is not None:
            target = _ReadTarget("user", str(uid), uid, raw_query.strip())
            return target

    if len(path_parts) >= 2 and path_parts[0] == "read":
        article_match = _ARTICLE_ID_RE.match(path_parts[1])
        if article_match is not None:
            numeric_id = int(article_match.group(1))
            target = _ReadTarget(
                "article",
                f"cv{numeric_id}",
                numeric_id,
                raw_query.strip(),
            )
            return target

    if len(path_parts) >= 2 and path_parts[0] == "live":
        room_id = _int_or_none(path_parts[1])
        if room_id is not None:
            target = _ReadTarget("live", str(room_id), room_id, raw_query.strip())
            return target

    if len(path_parts) >= 2 and path_parts[0] == "space":
        uid = _int_or_none(path_parts[1])
        if uid is not None:
            target = _ReadTarget("user", str(uid), uid, raw_query.strip())
            return target

    if len(path_parts) >= 3 and path_parts[0] == "bangumi":
        prefixed_id = _parse_prefixed_numeric_id(path_parts[2])
        if prefixed_id is not None:
            public_id = f"{prefixed_id[0]}{prefixed_id[1]}"
            target = _ReadTarget(
                "bangumi",
                public_id,
                prefixed_id[1],
                raw_query.strip(),
            )
            return target

    if len(path_parts) >= 2 and path_parts[0] in ("opus", "dynamic"):
        dynamic_id = _int_or_none(path_parts[1])
        if dynamic_id is not None:
            target = _ReadTarget(
                "dynamic",
                str(dynamic_id),
                dynamic_id,
                raw_query.strip(),
            )
            return target

    if len(path_parts) >= 2 and path_parts[0] == "audio":
        audio_match = _AUDIO_ID_RE.match(path_parts[1])
        if audio_match is not None:
            numeric_id = int(audio_match.group(1))
            target = _ReadTarget(
                "audio",
                f"au{numeric_id}",
                numeric_id,
                raw_query.strip(),
            )
            return target

    query_params = parse_qs(parsed_url.query)
    topic_ids = query_params.get("topic_id") or query_params.get("id")
    if "topic" in path_parts and topic_ids:
        topic_id = _int_or_none(topic_ids[0])
        if topic_id is not None:
            target = _ReadTarget(
                "topic",
                str(topic_id),
                topic_id,
                raw_query.strip(),
            )
            return target

    target = _ReadTarget("unknown", None, None, raw_query.strip())
    return target


def _parse_read_target(raw_query: str) -> _ReadTarget:
    """Parse a read query into one Bilibili content target."""
    stripped_query = raw_query.strip()
    url_target = _parse_bilibili_url(stripped_query)
    if url_target is not None:
        return url_target

    video_target = _read_target_from_video_id(stripped_query)
    if video_target is not None:
        return video_target

    target = _ReadTarget("unknown", None, None, None)
    return target


def _unsupported_result(
    decision: _RouterDecision,
    target: _ReadTarget,
    *,
    message: str,
) -> dict[str, Any]:
    """Build a bounded unsupported-target observation."""
    result = {
        "status": "unsupported",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "content_type": target.content_type,
        "content_scope": target.content_type,
        "public_id": target.public_id,
        "url": _public_url_for_target(target),
        "message": message,
    }
    return result


def _error_result(
    decision: _RouterDecision,
    *,
    content_type: str,
    public_id: str | None,
    url: str | None,
    message: str,
) -> dict[str, Any]:
    """Build a bounded source error observation."""
    result = {
        "status": "error",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "content_type": content_type,
        "content_scope": content_type,
        "public_id": public_id,
        "url": url,
        "message": message,
    }
    return result


async def _maybe_await(raw_value: object) -> object:
    """Await SDK coroutine results while accepting synchronous test fakes."""
    if inspect.isawaitable(raw_value):
        awaited_value = await raw_value
        return awaited_value

    return_value = raw_value
    return return_value


async def _call_object_method(obj: object, method_name: str, **kwargs: Any) -> object:
    """Call one SDK method and await the result when needed."""
    method = getattr(obj, method_name, None)
    if not callable(method):
        raise AttributeError(method_name)

    raw_value = method(**kwargs)
    result = await _maybe_await(raw_value)
    return result


def _compact_pages(raw_pages: object) -> list[dict[str, Any]]:
    """Compact Bilibili video page metadata."""
    if not isinstance(raw_pages, list):
        return_value: list[dict[str, Any]] = []
        return return_value

    pages: list[dict[str, Any]] = []
    for raw_page in raw_pages:
        if not isinstance(raw_page, dict):
            continue

        cid = _int_or_none(raw_page.get("cid"))
        page_number = _int_or_none(raw_page.get("page"))
        title = _first_text(raw_page, ("part", "title"))
        if cid is None and page_number is None and title is None:
            continue

        page: dict[str, Any] = {}
        if cid is not None:
            page["cid"] = cid
        if page_number is not None:
            page["page"] = page_number
        if title is not None:
            page["title"] = title
        pages.append(page)
        if len(pages) >= _PAGE_LIMIT:
            break

    return pages


def _extract_subtitle_url(raw_subtitle_payload: object) -> str | None:
    """Extract one subtitle body URL from SDK subtitle metadata."""
    if not isinstance(raw_subtitle_payload, dict):
        return_value = None
        return return_value

    raw_rows = None
    for field_name in ("subtitles", "subtitle", "list"):
        raw_candidate = raw_subtitle_payload.get(field_name)
        if isinstance(raw_candidate, list):
            raw_rows = raw_candidate
            break

    if raw_rows is None:
        nested_subtitle = raw_subtitle_payload.get("subtitle_info")
        if isinstance(nested_subtitle, dict):
            return_value = _extract_subtitle_url(nested_subtitle)
            return return_value

    if not isinstance(raw_rows, list):
        return_value = None
        return return_value

    for raw_row in raw_rows:
        if not isinstance(raw_row, dict):
            continue

        subtitle_url = _first_text(raw_row, ("subtitle_url", "url"))
        if subtitle_url is None:
            continue
        if subtitle_url.startswith("//"):
            subtitle_url = f"https:{subtitle_url}"

        return subtitle_url

    return_value = None
    return return_value


def _subtitle_excerpt_from_payload(raw_payload: object) -> str | None:
    """Compact subtitle body JSON into a bounded text excerpt."""
    if not isinstance(raw_payload, dict):
        return_value = None
        return return_value

    raw_body = raw_payload.get("body")
    if not isinstance(raw_body, list):
        return_value = None
        return return_value

    lines: list[str] = []
    for raw_line in raw_body:
        if not isinstance(raw_line, dict):
            continue

        content = _text_or_none(raw_line.get("content"))
        if content is None:
            continue

        lines.append(content)
        if sum(len(line) for line in lines) >= _SUBTITLE_CHAR_LIMIT:
            break

    if not lines:
        return_value = None
        return return_value

    excerpt = "\n".join(lines)
    return_value = excerpt[:_SUBTITLE_CHAR_LIMIT]
    return return_value


async def _fetch_subtitle_excerpt(subtitle_url: str) -> str | None:
    """Fetch subtitle JSON body from a bounded URL returned by the SDK."""
    try:
        async with httpx.AsyncClient(timeout=_SUBTITLE_TIMEOUT_SECONDS) as client:
            response = await client.get(subtitle_url)
            response.raise_for_status()
            payload = response.json()
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return_value = None
        return return_value

    result = _subtitle_excerpt_from_payload(payload)
    return result


async def _read_video(
    decision: _RouterDecision,
    target: _ReadTarget,
) -> dict[str, Any]:
    """Read one public Bilibili video by BV or av id."""
    video_module = importlib.import_module("bilibili_api.video")
    video_class = getattr(video_module, "Video")
    if target.numeric_id is None:
        video = video_class(bvid=target.public_id)
    else:
        video = video_class(aid=target.numeric_id)

    raw_info = await _call_object_method(video, "get_info")
    if not isinstance(raw_info, dict):
        result = _error_result(
            decision,
            content_type="video",
            public_id=target.public_id,
            url=_public_url_for_target(target),
            message="Unexpected Bilibili video metadata response shape.",
        )
        return result

    raw_pages = await _call_object_method(video, "get_pages")
    pages = _compact_pages(raw_pages)
    subtitle_excerpt = None
    if pages and "cid" in pages[0]:
        try:
            raw_subtitle = await _call_object_method(
                video,
                "get_subtitle",
                cid=pages[0]["cid"],
            )
        except _external_bilibili_error_types():
            raw_subtitle = None

        if raw_subtitle is not None:
            subtitle_url = _extract_subtitle_url(raw_subtitle)
            if subtitle_url is not None:
                subtitle_excerpt = await _fetch_subtitle_excerpt(subtitle_url)

    content_basis = ["metadata"]
    if pages:
        content_basis.append("pages")
    if subtitle_excerpt is not None:
        content_basis.append("subtitle")

    result = {
        "status": "success",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "content_type": "video",
        "content_scope": "video",
        "public_id": target.public_id,
        "url": _public_url_for_target(target),
        "title": _first_text(raw_info, ("title", "name")),
        "creator": _creator_from_payload(raw_info),
        "summary": _first_text(raw_info, ("desc", "description", "summary")),
        "stats_summary": _compact_stats(_stats_payload(raw_info)),
        "duration_seconds": _int_or_none(raw_info.get("duration")),
        "pages": pages,
        "subtitle_excerpt": subtitle_excerpt,
        "content_basis": content_basis,
        "message": "Bilibili video metadata loaded.",
    }
    return result


def _protocols_for_target(target: _ReadTarget) -> tuple[_SdkReadProtocol, ...]:
    """Return best-effort SDK read protocols for one non-video target."""
    numeric_id = target.numeric_id
    if numeric_id is None:
        return_value: tuple[_SdkReadProtocol, ...] = ()
        return return_value

    if target.content_type == "article":
        return_value = (
            _SdkReadProtocol(
                module_name="bilibili_api.article",
                class_names=("Article",),
                kwargs_options=({"cvid": numeric_id}, {"id": numeric_id}),
                method_names=("get_info", "get_content"),
            ),
        )
        return return_value

    if target.content_type == "live":
        return_value = (
            _SdkReadProtocol(
                module_name="bilibili_api.live",
                class_names=("LiveRoom",),
                kwargs_options=(
                    {"room_display_id": numeric_id},
                    {"room_id": numeric_id},
                    {"id": numeric_id},
                ),
                method_names=("get_room_info", "get_info"),
            ),
        )
        return return_value

    if target.content_type == "user":
        return_value = (
            _SdkReadProtocol(
                module_name="bilibili_api.user",
                class_names=("User",),
                kwargs_options=({"uid": numeric_id}, {"mid": numeric_id}),
                method_names=("get_user_info", "get_info"),
            ),
        )
        return return_value

    if target.content_type == "bangumi":
        id_prefix = ""
        if target.public_id is not None:
            parsed_id = _parse_prefixed_numeric_id(target.public_id)
            if parsed_id is not None:
                id_prefix = parsed_id[0]
        kwargs_options = [{"id": numeric_id}]
        if id_prefix == "ep":
            kwargs_options.insert(0, {"epid": numeric_id})
        if id_prefix == "ss":
            kwargs_options.insert(0, {"season_id": numeric_id})
        if id_prefix == "md":
            kwargs_options.insert(0, {"media_id": numeric_id})

        return_value = (
            _SdkReadProtocol(
                module_name="bilibili_api.bangumi",
                class_names=("Bangumi", "Episode", "Media"),
                kwargs_options=tuple(kwargs_options),
                method_names=("get_info", "get_meta"),
            ),
        )
        return return_value

    if target.content_type == "dynamic":
        return_value = (
            _SdkReadProtocol(
                module_name="bilibili_api.dynamic",
                class_names=("Dynamic",),
                kwargs_options=({"dynamic_id": numeric_id}, {"id": numeric_id}),
                method_names=("get_info", "get_detail"),
            ),
            _SdkReadProtocol(
                module_name="bilibili_api.opus",
                class_names=("Opus",),
                kwargs_options=({"opus_id": numeric_id}, {"id": numeric_id}),
                method_names=("get_info", "get_detail"),
            ),
        )
        return return_value

    if target.content_type == "audio":
        return_value = (
            _SdkReadProtocol(
                module_name="bilibili_api.audio",
                class_names=("Audio",),
                kwargs_options=({"auid": numeric_id}, {"audio_id": numeric_id}),
                method_names=("get_info", "get_detail"),
            ),
        )
        return return_value

    if target.content_type == "topic":
        return_value = (
            _SdkReadProtocol(
                module_name="bilibili_api.topic",
                class_names=("Topic",),
                kwargs_options=({"topic_id": numeric_id}, {"id": numeric_id}),
                method_names=("get_info", "get_detail"),
            ),
        )
        return return_value

    return_value = ()
    return return_value


def _instantiate_protocol(protocol: _SdkReadProtocol) -> object | None:
    """Instantiate the first available SDK class for a generic read protocol."""
    try:
        sdk_module = importlib.import_module(protocol.module_name)
    except ImportError:
        instance = None
        return instance

    for class_name in protocol.class_names:
        sdk_class = getattr(sdk_module, class_name, None)
        if sdk_class is None:
            continue
        if not callable(sdk_class):
            continue

        for kwargs in protocol.kwargs_options:
            try:
                instance = sdk_class(**kwargs)
            except TypeError:
                continue

            return instance

    instance = None
    return instance


async def _read_with_protocol(
    protocol: _SdkReadProtocol,
) -> object | None:
    """Read one non-video target through a best-effort SDK protocol."""
    instance = _instantiate_protocol(protocol)
    if instance is None:
        return_value = None
        return return_value

    for method_name in protocol.method_names:
        method = getattr(instance, method_name, None)
        if not callable(method):
            continue

        raw_value = method()
        result = await _maybe_await(raw_value)
        return result

    return_value = None
    return return_value


def _compact_generic_read_payload(
    decision: _RouterDecision,
    target: _ReadTarget,
    payload: object,
) -> dict[str, Any]:
    """Compact non-video read payloads to prompt-safe public metadata."""
    if isinstance(payload, dict):
        title = _first_text(payload, ("title", "name", "season_title"))
        creator = _creator_from_payload(payload)
        summary = _first_text(
            payload,
            ("summary", "desc", "description", "intro", "content"),
        )
        stats_summary = _compact_stats(_stats_payload(payload))
    else:
        title = None
        creator = None
        summary = _text_or_none(payload)
        stats_summary = None

    result = {
        "status": "success",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "content_type": target.content_type,
        "content_scope": target.content_type,
        "public_id": target.public_id,
        "url": _public_url_for_target(target),
        "title": title,
        "creator": creator,
        "summary": summary,
        "stats_summary": stats_summary,
        "content_basis": ["metadata"],
        "message": f"Bilibili {target.content_type} metadata loaded.",
    }
    return result


async def _read_generic_target(
    decision: _RouterDecision,
    target: _ReadTarget,
) -> dict[str, Any]:
    """Read a non-video Bilibili target when the SDK exposes a stable protocol."""
    for protocol in _protocols_for_target(target):
        payload = await _read_with_protocol(protocol)
        if payload is None:
            continue

        result = _compact_generic_read_payload(decision, target, payload)
        return result

    result = _unsupported_result(
        decision,
        target,
        message="Bilibili SDK does not expose a supported read API for this target.",
    )
    return result


async def _read_bilibili(decision: _RouterDecision) -> dict[str, Any]:
    """Execute a Bilibili read decision with source-local target parsing."""
    target = _parse_read_target(decision.query)
    if target.content_type == "unknown":
        result = _unsupported_result(
            decision,
            target,
            message="Unsupported or unrecognized Bilibili public target.",
        )
        return result

    if target.content_type not in _READ_CONTENT_TYPES:
        result = _unsupported_result(
            decision,
            target,
            message="Unsupported or unrecognized Bilibili public target.",
        )
        return result

    try:
        _load_sdk_root()
        if target.content_type == "video":
            result = await _read_video(decision, target)
            return result

        result = await _read_generic_target(decision, target)
        return result
    except _external_bilibili_error_types() as exc:
        result = _error_result(
            decision,
            content_type=target.content_type,
            public_id=target.public_id,
            url=_public_url_for_target(target),
            message=f"Bilibili read failed: {exc}",
        )
        return result


def _infer_search_scope(query: str) -> str:
    """Infer provider search scope from the semantic user query."""
    lowered_query = query.lower()
    scope_markers = (
        ('video', ('视频', 'video', 'bv', 'av')),
        ('article', ('专栏', '文章', 'article', 'read')),
        ('bangumi', ('番剧', '影视', '电影', 'bangumi', 'anime')),
        ('live', ('直播', 'live')),
        ('user', ('up主', 'up 主', '用户', 'user', 'space')),
        ('dynamic', ('动态', 'opus', 'dynamic')),
        ('audio', ('音频', '音乐', 'audio')),
        ('topic', ('话题', 'topic')),
    )
    for scope, markers in scope_markers:
        for marker in markers:
            if marker in lowered_query:
                return scope

    has_popular_marker = any(marker in lowered_query for marker in _POPULAR_MARKERS)
    if has_popular_marker:
        return_value = "video"
        return return_value

    return_value = "general"
    return return_value


def _infer_popularity_basis(query: str, scope: str) -> str:
    """Infer the requested popularity ordering semantics."""
    lowered_query = query.lower()
    has_popular_marker = any(marker in lowered_query for marker in _POPULAR_MARKERS)
    if scope == "video" and has_popular_marker:
        return_value = "most_clicked"
        return return_value

    if scope == "video":
        return_value = "comprehensive"
        return return_value

    return_value = "provider_default"
    return return_value


async def _call_search_function(
    search_function: object,
    query: str,
    **kwargs: Any,
) -> object:
    """Call an SDK search function with async/sync compatibility."""
    if not callable(search_function):
        raise AttributeError("search function unavailable")

    raw_value = search_function(query, **kwargs)
    result = await _maybe_await(raw_value)
    return result


def _search_rows_from_payload(payload: object) -> list[object]:
    """Extract provider result rows from common search response shapes."""
    if isinstance(payload, list):
        return_value = payload
        return return_value

    if not isinstance(payload, dict):
        return_value = []
        return return_value

    raw_result = payload.get("result")
    if isinstance(raw_result, list):
        flattened_rows: list[object] = []
        for raw_row in raw_result:
            if isinstance(raw_row, dict) and isinstance(raw_row.get("data"), list):
                flattened_rows.extend(raw_row["data"])
                continue

            flattened_rows.append(raw_row)

        return_value = flattened_rows
        return return_value

    raw_data = payload.get("data")
    if isinstance(raw_data, list):
        return_value = raw_data
        return return_value

    if isinstance(raw_data, dict):
        nested_result = raw_data.get("result")
        if isinstance(nested_result, list):
            return_value = nested_result
            return return_value

    return_value = []
    return return_value


def _public_id_from_url(raw_url: str) -> str | None:
    """Extract a stable public id from a Bilibili URL."""
    target = _parse_read_target(raw_url)
    public_id = target.public_id
    return public_id


def _compact_search_row(
    raw_row: object,
    *,
    fallback_scope: str,
) -> dict[str, Any] | None:
    """Compact one Bilibili search row to stable candidate fields."""
    if not isinstance(raw_row, dict):
        compact_row = None
        return compact_row

    title = _first_text(raw_row, ("title", "name"))
    url = _first_text(raw_row, ("arcurl", "url", "link"))
    public_id = _first_text(raw_row, ("bvid", "season_id", "media_id"))
    if public_id is None and url is not None:
        public_id = _public_id_from_url(url)
    if url is None and public_id is not None and public_id.startswith("BV"):
        url = f"{_BILIBILI_PUBLIC_BASE_URL}/video/{public_id}/"
    if title is None or url is None:
        compact_row = None
        return compact_row

    row_scope = _first_text(raw_row, ("type", "result_type"))
    if row_scope is None:
        row_scope = fallback_scope

    compact_row = {
        "title": title,
        "url": url,
        "content_type": row_scope,
        "public_id": public_id,
        "creator": _creator_from_payload(raw_row),
        "summary": _first_text(raw_row, ("description", "desc", "content")),
        "stats_summary": _compact_stats(_stats_payload(raw_row)),
    }
    return compact_row


def _compact_search_results(payload: object, *, scope: str) -> list[dict[str, Any]]:
    """Compact and cap Bilibili search candidates."""
    rows = _search_rows_from_payload(payload)
    results: list[dict[str, Any]] = []
    fallback_scope = "mixed" if scope == "general" else scope
    for raw_row in rows:
        compact_row = _compact_search_row(raw_row, fallback_scope=fallback_scope)
        if compact_row is None:
            continue

        results.append(compact_row)
        if len(results) >= _SEARCH_RESULT_LIMIT:
            break

    return results


def _typed_search_kwargs(
    search_module: ModuleType,
    scope: str,
    popularity_basis: str,
) -> dict[str, Any] | None:
    """Build SDK kwargs for a typed search scope."""
    search_object_type = getattr(search_module, "SearchObjectType", None)
    enum_name = _SEARCH_SCOPE_ENUM_NAMES.get(scope)
    if enum_name is None or search_object_type is None:
        kwargs = None
        return kwargs

    search_type = getattr(search_object_type, enum_name, None)
    if search_type is None:
        kwargs = None
        return kwargs

    kwargs = {
        "search_type": search_type,
        "page": _SEARCH_PAGE,
    }
    if scope == "video":
        order_video = getattr(search_module, "OrderVideo", None)
        if order_video is not None and popularity_basis == "most_clicked":
            order_type = getattr(order_video, "CLICK", None)
            if order_type is not None:
                kwargs["order_type"] = order_type
        elif order_video is not None:
            order_type = getattr(order_video, "TOTALRANK", None)
            if order_type is not None:
                kwargs["order_type"] = order_type

    return kwargs


async def _search_bilibili(decision: _RouterDecision) -> dict[str, Any]:
    """Execute a Bilibili semantic search decision."""
    scope = _infer_search_scope(decision.query)
    popularity_basis = _infer_popularity_basis(decision.query, scope)
    try:
        _load_sdk_root()
        search_module = importlib.import_module("bilibili_api.search")
        if scope == "general":
            search_function = getattr(search_module, "search", None)
            payload = await _call_search_function(
                search_function,
                decision.query,
                page=_SEARCH_PAGE,
            )
        else:
            search_function = getattr(search_module, "search_by_type", None)
            kwargs = _typed_search_kwargs(search_module, scope, popularity_basis)
            if kwargs is None:
                target = _ReadTarget(scope, None, None, None)
                result = _unsupported_result(
                    decision,
                    target,
                    message="Bilibili SDK does not expose a supported search scope.",
                )
                return result

            payload = await _call_search_function(
                search_function,
                decision.query,
                **kwargs,
            )
    except _external_bilibili_error_types() as exc:
        result = _error_result(
            decision,
            content_type=scope,
            public_id=None,
            url=None,
            message=f"Bilibili search failed: {exc}",
        )
        return result

    compact_results = _compact_search_results(payload, scope=scope)
    result = {
        "status": "success",
        "source": decision.source,
        "action": decision.action,
        "query": decision.query,
        "content_scope": scope,
        "popularity_basis": popularity_basis,
        "message": "Bilibili search completed.",
        "results": compact_results,
    }
    return result


async def execute(decision: _RouterDecision) -> dict[str, Any]:
    """Execute Bilibili read/search decisions with source-local parsing."""
    if decision.action == "stop":
        result = {
            "status": "stopped",
            "source": decision.source,
            "action": decision.action,
            "query": decision.query,
            "message": "Router stopped without another Bilibili action.",
        }
        return result

    if decision.action == "read":
        result = await _read_bilibili(decision)
        return result

    result = await _search_bilibili(decision)
    return result
