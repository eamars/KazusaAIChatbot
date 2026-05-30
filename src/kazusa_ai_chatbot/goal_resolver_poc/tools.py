"""Bounded tool adapters for the goal resolver POC."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus, urljoin

import httpx
from mcp.shared.exceptions import McpError

from kazusa_ai_chatbot.goal_resolver_poc.llm import (
    call_sandbox_patcher,
    call_self_goal_generator,
)
from kazusa_ai_chatbot.goal_resolver_poc.models import bounded_text
from kazusa_ai_chatbot.nodes.persona_supervisor2_rag_supervisor2 import (
    call_rag_supervisor,
)
from kazusa_ai_chatbot.rag.web_agent3 import WebAgent3
import kazusa_ai_chatbot.rag.web_agent3.searxng_tools as searxng_tools


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_ROOT = REPO_ROOT / "resources" / "goal_resolver_poc" / "fixtures"
CODE_REPAIR_FIXTURE_ROOT = FIXTURE_ROOT / "code_repair"
INCIDENT_LOG_FIXTURE_ROOT = FIXTURE_ROOT / "incident_logs"
COMMAND_TIMEOUT_SECONDS = 20
MAX_DIRECTORY_FILES = 24
MAX_FILE_CHARS = 5000
PUBLIC_SEARCH_MAX_CHARS = 6000
PUBLIC_SEARCH_RESULT_LIMIT = 8
PUBLIC_SEARCH_TIMEOUT_SECONDS = 12
PUBLIC_CATALOG_TIMEOUT_SECONDS = 12
PUBLIC_PAGE_READ_LIMIT = 2
PUBLIC_PAGE_EXCERPT_CHARS = 3000
PUBLIC_CATALOG_MAX_CHARS = 12000
PUBLIC_CATALOG_TERM_LIMIT = 8
PUBLIC_CATALOG_PRODUCTS_PER_TERM = 4
PUBLIC_RESTAURANT_DIRECTORY_MAX_ROWS = 4
PUBLIC_GITHUB_RELEASE_MAX_ROWS = 5
FOODGUIDE_AUCKLAND_SEARCH_URL = "https://foodguide.nz/auckland/?s=japanese"
PUBLIC_SEARCH_HEADERS = {
    "User-Agent": "Mozilla/5.0 KazusaGoalResolverPOC/1.0",
}
NON_PRODUCT_CODES = frozenset({"FP16", "FP32", "BF16", "INT8", "INT4"})
HIGH_VRAM_CATALOG_TERMS = (
    "32GB graphics card",
    "24GB graphics card",
    "workstation graphics card",
    "NVIDIA graphics card",
)
HIGH_END_COMPONENT_CATALOG_TERMS = (
    "Ryzen 9 processor",
    "Intel Core i9 processor",
    "64GB DDR5 memory",
    "2TB NVMe SSD",
    "1000W ATX 3.0 PSU",
    "airflow PC case",
)
READY_SYSTEM_MARKERS = (
    "desktop",
    "gaming pc",
    " pc",
    "pc ",
    "ready to ship",
    "workstation pc",
)
GRAPHICS_PRODUCT_MARKERS = (
    "graphics card",
    "gpu",
    "geforce rtx",
)
COMPONENT_PRODUCT_MARKERS = (
    "case",
    "cpu",
    "memory",
    "nvme",
    "power supply",
    "processor",
    "psu",
    "ssd",
)
_PRODUCT_CODE_RE = re.compile(
    r"\b(?:RTX|GTX)\s+\d{4}(?:\s+(?:SUPER|TI))?\b|\b[A-Z]{1,5}\d{2,4}S?\b",
    flags=re.IGNORECASE,
)
_PUBLIC_SEARCH_URL_TEMPLATE = "https://www.bing.com/search?format=rss&q={query}"
_PUBLIC_PRODUCT_CATALOGS = (
    {
        "name": "Computer Lounge product suggest",
        "base_url": "https://computerlounge.co.nz",
        "url_template": (
            "https://computerlounge.co.nz/search/suggest.json?"
            "q={query}&resources[type]=product"
        ),
    },
)
_QUERY_STOPWORDS = frozenset({
    "and",
    "are",
    "for",
    "from",
    "guide",
    "how",
    "list",
    "model",
    "models",
    "new",
    "or",
    "requirements",
    "search",
    "the",
    "usage",
    "with",
})


class _VisibleTextParser(HTMLParser):
    """Extract readable text from HTML for bounded public evidence excerpts."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        del attrs
        if tag.lower() in {"script", "style", "svg", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "svg", "noscript"}:
            self._skip_depth = max(0, self._skip_depth - 1)

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        stripped = data.strip()
        if stripped:
            self.parts.append(stripped)


def _resolve_inside(path: Path, root: Path) -> Path:
    """Resolve `path` and require that it stays inside `root`.

    Args:
        path: Candidate filesystem path.
        root: Directory boundary the candidate must stay within.

    Returns:
        The resolved candidate path.

    Raises:
        ValueError: If the path escapes the root directory.
    """

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(
            f"path {resolved_path} escapes allowed root {resolved_root}"
        ) from exc
    return resolved_path


def _normalize_sandbox_relative_path(path_text: str) -> Path:
    """Normalize an LLM patch path into the code sandbox namespace."""

    normalized_text = path_text.strip().replace("\\", "/")
    marker = "code_repair/"
    if marker in normalized_text:
        normalized_text = normalized_text.rsplit(marker, maxsplit=1)[1]
    candidate = Path(normalized_text)
    if candidate.is_absolute():
        candidate = Path(candidate.name)
    if any(part == ".." for part in candidate.parts):
        raise ValueError(f"sandbox patch path contains parent traversal: {path_text}")
    return candidate


def _safe_file_text(path: Path) -> str:
    """Read one text file with a bounded artifact payload."""

    text = path.read_text(encoding="utf-8")
    return_value = bounded_text(text, limit=MAX_FILE_CHARS)
    return return_value


def _directory_snapshot(root: Path) -> list[dict[str, str]]:
    """Read a small text snapshot from a fixture or sandbox directory."""

    resolved_root = root.resolve()
    entries: list[dict[str, str]] = []
    for path in sorted(resolved_root.rglob("*")):
        if len(entries) >= MAX_DIRECTORY_FILES:
            break
        if not path.is_file():
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        relative_path = path.relative_to(resolved_root).as_posix()
        try:
            content = _safe_file_text(path)
        except UnicodeDecodeError as exc:
            content = f"unreadable text file: {exc}"
        entries.append({"path": relative_path, "content": content})
    return entries


def _code_sandbox_root(state: dict[str, Any]) -> Path:
    """Return the sandbox root for local code-repair actions."""

    sandbox_root = Path(state["sandbox_root"]) / "code_repair"
    return sandbox_root


def _ensure_code_sandbox(state: dict[str, Any]) -> Path:
    """Create an isolated code-repair sandbox if it is not present."""

    sandbox_root = _code_sandbox_root(state)
    if sandbox_root.exists():
        return sandbox_root
    if not CODE_REPAIR_FIXTURE_ROOT.exists():
        raise FileNotFoundError(
            f"missing fixture directory: {CODE_REPAIR_FIXTURE_ROOT}"
        )
    sandbox_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(CODE_REPAIR_FIXTURE_ROOT, sandbox_root)
    return sandbox_root


def _observation(
    *,
    observation_id: str,
    tool: str,
    target_requirement_id: str,
    status: str,
    summary: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Build one serializable tool observation row."""

    row = {
        "observation_id": observation_id,
        "tool": tool,
        "target_requirement_id": target_requirement_id,
        "status": status,
        "summary": summary,
        "payload": payload,
    }
    return row


def _catalog_search_terms(query: str) -> list[str]:
    """Extract bounded product-code search terms from a public-web query."""

    terms: list[str] = []
    for match in _PRODUCT_CODE_RE.finditer(query):
        term = " ".join(match.group(0).upper().split())
        if term in NON_PRODUCT_CODES:
            continue
        if term not in terms:
            terms.append(term)

    lower_query = query.lower()
    is_graphics_query = (
        "graphics card" in lower_query
        or "gpu" in lower_query
        or "显卡" in query
    )
    if is_graphics_query:
        enriched_terms = [
            f"{term} graphics card"
            for term in terms
            if "RTX" in term or "GTX" in term
        ]
        ordered_terms = list(enriched_terms)
        for term in terms:
            if term not in ordered_terms:
                ordered_terms.append(term)
        terms = ordered_terms

    is_new_zealand_query = (
        "new zealand" in lower_query
        or "nz" in lower_query
        or "新西兰" in query
    )
    is_hardware_query = (
        is_graphics_query
        or "vram" in lower_query
        or "hardware" in lower_query
        or "硬件" in query
    )
    query_tokens = _query_tokens(query)
    component_tokens = {
        "case",
        "component",
        "components",
        "cpu",
        "ddr5",
        "memory",
        "nvme",
        "psu",
        "ram",
        "ssd",
    }
    is_component_query = bool(query_tokens.intersection(component_tokens))
    if is_new_zealand_query and is_component_query:
        ordered_terms = list(HIGH_END_COMPONENT_CATALOG_TERMS)
        for term in terms:
            if term not in ordered_terms:
                ordered_terms.append(term)
        terms = ordered_terms

    if is_new_zealand_query and is_hardware_query:
        for term in HIGH_VRAM_CATALOG_TERMS:
            if term not in terms:
                terms.append(term)

    return_value = terms[:PUBLIC_CATALOG_TERM_LIMIT]
    return return_value


def _query_tokens(query: str) -> set[str]:
    """Extract relevance tokens from a public-web query."""

    tokens = set()
    for match in re.finditer(r"[A-Za-z0-9.]+", query.lower()):
        token = match.group(0)
        if len(token) < 2 or token in _QUERY_STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _search_item_score(item: dict[str, str], tokens: set[str]) -> int:
    """Score one public search item for page-read ordering."""

    title = item["title"].lower()
    description = item["description"].lower()
    combined = f"{title} {description}"
    score = 0
    for token in tokens:
        if token in title:
            score += 3
        elif token in combined:
            score += 1
    if "official" in combined or "github" in combined:
        score += 1
    return score


def _has_relevant_public_search_items(
    query: str,
    public_search_items: dict[str, Any],
) -> bool:
    """Return true when parsed public search rows overlap the query."""

    items = public_search_items.get("items")
    if not isinstance(items, list):
        return False
    tokens = _query_tokens(query)
    for item in items:
        if not isinstance(item, dict):
            continue
        score = _search_item_score(item, tokens)
        if score >= 3:
            return True
    return False


def _looks_like_model_requirement_query(query: str) -> bool:
    """Return true for hardware-requirement evidence rather than retail stock."""

    lower_query = query.lower()
    result = any(
        marker in lower_query
        for marker in {
            "model size",
            "quantization",
            "requirements",
            "vram",
        }
    ) or any(marker in query for marker in {"显存", "量化", "需求"})
    return result


def _html_to_visible_text(raw_html: str) -> str:
    """Convert HTML into normalized visible text."""

    parser = _VisibleTextParser()
    parser.feed(raw_html)
    joined = " ".join(parser.parts)
    text = re.sub(r"\s+", " ", joined).strip()
    return text


def _strip_html_fragment(raw_html: str) -> str:
    """Convert a small HTML fragment into plain visible text."""

    return _html_to_visible_text(raw_html)


def _focused_excerpt(text: str, query: str) -> str:
    """Return a bounded excerpt near query-relevant terms."""

    lower_text = text.lower()
    tokens = _query_tokens(query)
    anchors = [
        "q4_k_m",
        "q4",
        "vram",
        " gb",
        "rtx",
        "release",
    ] + list(tokens)
    start_index = -1
    for anchor in anchors:
        found_index = lower_text.find(anchor)
        if found_index < 0:
            continue
        start_index = found_index
        break
    if start_index < 0:
        excerpt = text[:PUBLIC_PAGE_EXCERPT_CHARS]
        return excerpt
    window_start = max(0, start_index - 700)
    window_end = window_start + PUBLIC_PAGE_EXCERPT_CHARS
    excerpt = text[window_start:window_end]
    return excerpt


async def _read_public_url(url: str, max_length: int) -> str:
    """Read one public URL through the configured web URL reader."""

    result = await searxng_tools.web_url_read.ainvoke(
        {"url": url, "maxLength": max_length}
    )
    return_value = str(result)
    return return_value


async def _fetch_public_url_text(url: str, timeout_seconds: int) -> str:
    """Fetch one public text URL through a bounded HTTP client."""

    timeout = httpx.Timeout(timeout_seconds)
    async with httpx.AsyncClient(
        headers=PUBLIC_SEARCH_HEADERS,
        timeout=timeout,
        follow_redirects=True,
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
    text = response.text
    return text


async def _public_search_fallback(query: str) -> dict[str, str]:
    """Read a public RSS search result page when direct search returns empty."""

    url = _PUBLIC_SEARCH_URL_TEMPLATE.format(query=quote_plus(query))
    try:
        result = await _read_public_url(url, PUBLIC_SEARCH_MAX_CHARS)
    except (McpError, OSError, TimeoutError) as exc:
        fallback = {
            "status": "error",
            "url": url,
            "text": "",
            "error": f"public RSS search fallback failed: {exc}",
        }
        return fallback

    status = "observed"
    if not result.strip():
        status = "empty"
    fallback = {
        "status": status,
        "url": url,
        "text": result,
        "error": "",
    }
    return fallback


async def _public_search_items(query: str) -> dict[str, Any]:
    """Fetch and parse public RSS search rows into compact evidence items."""

    url = _PUBLIC_SEARCH_URL_TEMPLATE.format(query=quote_plus(query))
    try:
        response_text = await _fetch_public_url_text(
            url,
            PUBLIC_SEARCH_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        result = {
            "url": url,
            "items": [],
            "error": f"public RSS item fetch failed: {exc}",
        }
        return result

    try:
        root = ET.fromstring(response_text)
    except ET.ParseError as exc:
        result = {
            "url": url,
            "items": [],
            "error": f"public RSS item parse failed: {exc}",
        }
        return result

    items: list[dict[str, str]] = []
    for item in root.findall("./channel/item")[:PUBLIC_SEARCH_RESULT_LIMIT]:
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        description = (item.findtext("description") or "").strip()
        published_at = (item.findtext("pubDate") or "").strip()
        search_item = {
            "title": bounded_text(title, limit=240),
            "url": bounded_text(link, limit=500),
            "description": bounded_text(description, limit=700),
            "published_at": bounded_text(published_at, limit=120),
        }
        items.append(search_item)

    result = {
        "url": url,
        "items": items,
        "error": "",
    }
    return result


async def _public_page_excerpts(
    query: str,
    public_search_items: dict[str, Any],
) -> list[dict[str, str]]:
    """Read compact page excerpts from the most relevant search result URLs."""

    items = public_search_items.get("items")
    if not isinstance(items, list):
        return_value: list[dict[str, str]] = []
        return return_value

    tokens = _query_tokens(query)
    candidate_items = [
        item
        for item in items
        if isinstance(item, dict) and str(item.get("url", "")).startswith("http")
    ]
    scored_items = sorted(
        candidate_items,
        key=lambda item: _search_item_score(item, tokens),
        reverse=True,
    )
    excerpts: list[dict[str, str]] = []
    for item in scored_items[:PUBLIC_PAGE_READ_LIMIT]:
        url = str(item.get("url", ""))
        try:
            raw_html = await _fetch_public_url_text(
                url,
                PUBLIC_SEARCH_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            excerpts.append(
                {
                    "title": bounded_text(item.get("title", ""), limit=240),
                    "url": bounded_text(url, limit=500),
                    "error": f"public page read failed: {exc}",
                    "excerpt": "",
                }
            )
            continue
        visible_text = _html_to_visible_text(raw_html)
        excerpts.append(
            {
                "title": bounded_text(item.get("title", ""), limit=240),
                "url": bounded_text(url, limit=500),
                "error": "",
                "excerpt": bounded_text(
                    _focused_excerpt(visible_text, query),
                    limit=PUBLIC_PAGE_EXCERPT_CHARS,
                ),
            }
        )

    return excerpts


def _looks_like_auckland_japanese_restaurant_query(query: str) -> bool:
    """Return true when a query asks for Auckland Japanese dining evidence."""

    lower_query = query.lower()
    has_location = "auckland" in lower_query or "奥克兰" in query
    has_cuisine = "japanese" in lower_query or "日料" in query or "日本" in query
    has_dining = (
        "restaurant" in lower_query
        or "dining" in lower_query
        or "food" in lower_query
        or "reservation" in lower_query
        or "booking" in lower_query
        or "walk-in" in lower_query
        or "餐厅" in query
        or "吃" in query
        or "日料店" in query
    )
    return has_location and has_cuisine and has_dining


def _looks_like_github_release_query(query: str) -> bool:
    """Return true when a query asks for public software release evidence."""

    lower_query = query.lower()
    has_release = (
        "release" in lower_query
        or "正式发布" in query
        or "發布" in query
        or "发布" in query
    )
    has_code_source = (
        "github" in lower_query
        or "repository" in lower_query
        or "repo" in lower_query
        or "项目" in query
        or "project" in lower_query
    )
    return has_release and has_code_source


def _github_release_candidate_repos(query: str) -> list[str]:
    """Infer likely GitHub repositories from a release-date query."""

    repos: list[str] = []
    for match in re.finditer(
        r"github\.com/(?P<owner>[A-Za-z0-9_.-]+)/(?P<repo>[A-Za-z0-9_.-]+)",
        query,
        flags=re.IGNORECASE,
    ):
        repo = f"{match.group('owner')}/{match.group('repo')}"
        if repo not in repos:
            repos.append(repo)

    project_names = []
    for match in re.finditer(r"\b[A-Z][A-Za-z0-9]*(?:[A-Z][A-Za-z0-9]*)\b", query):
        name = match.group(0).strip()
        if len(name) < 4 or name in {"GitHub", "Release", "Releases"}:
            continue
        if name not in project_names:
            project_names.append(name)

    for project_name in project_names:
        candidate = f"{project_name}/{project_name}"
        if candidate not in repos:
            repos.append(candidate)
        if "all hands ai" in query.lower():
            candidate = f"All-Hands-AI/{project_name}"
            if candidate not in repos:
                repos.append(candidate)

    return repos[:PUBLIC_GITHUB_RELEASE_MAX_ROWS]


async def _github_release_fallback(query: str) -> list[dict[str, Any]]:
    """Read GitHub Releases directly for software release-date tasks."""

    if not _looks_like_github_release_query(query):
        return_value: list[dict[str, Any]] = []
        return return_value

    rows: list[dict[str, Any]] = []
    for repo in _github_release_candidate_repos(query):
        url = f"https://api.github.com/repos/{repo}/releases?per_page=10"
        try:
            response_text = await _fetch_public_url_text(
                url,
                PUBLIC_SEARCH_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            rows.append(
                {
                    "source": "GitHub Releases API",
                    "requested_repo": repo,
                    "url": url,
                    "error": f"github releases fetch failed: {exc}",
                    "releases": [],
                }
            )
            continue
        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError as exc:
            rows.append(
                {
                    "source": "GitHub Releases API",
                    "requested_repo": repo,
                    "url": url,
                    "error": f"github releases parse failed: {exc}",
                    "releases": [],
                }
            )
            continue
        if not isinstance(parsed, list):
            rows.append(
                {
                    "source": "GitHub Releases API",
                    "requested_repo": repo,
                    "url": url,
                    "error": "github releases response was not a list",
                    "releases": [],
                }
            )
            continue

        releases: list[dict[str, Any]] = []
        for release in parsed:
            if not isinstance(release, dict):
                continue
            if release.get("draft") or release.get("prerelease"):
                continue
            release_row = {
                "tag_name": bounded_text(release.get("tag_name", ""), limit=80),
                "name": bounded_text(release.get("name", ""), limit=160),
                "published_at": bounded_text(
                    release.get("published_at", ""),
                    limit=80,
                ),
                "created_at": bounded_text(release.get("created_at", ""), limit=80),
                "html_url": bounded_text(release.get("html_url", ""), limit=300),
                "draft": bool(release.get("draft")),
                "prerelease": bool(release.get("prerelease")),
            }
            releases.append(release_row)
            if len(releases) >= PUBLIC_GITHUB_RELEASE_MAX_ROWS:
                break
        rows.append(
            {
                "source": "GitHub Releases API",
                "requested_repo": repo,
                "url": url,
                "error": "",
                "releases": releases,
            }
        )
        if releases:
            break
    return rows


def _github_release_summary_text(rows: list[dict[str, Any]]) -> str:
    """Render GitHub release rows as concise evidence."""

    lines: list[str] = []
    for row in rows:
        if row.get("error"):
            lines.append(
                f"{row.get('source', '')} {row.get('requested_repo', '')}: "
                f"{row['error']}"
            )
            continue
        releases = row.get("releases")
        if not isinstance(releases, list):
            continue
        for release in releases[:3]:
            if not isinstance(release, dict):
                continue
            line = (
                f"{row.get('source', '')} repo={row.get('requested_repo', '')}: "
                f"tag={release.get('tag_name', '')}; "
                f"name={release.get('name', '')}; "
                f"published_at={release.get('published_at', '')}; "
                f"prerelease={release.get('prerelease')}; "
                f"draft={release.get('draft')}; "
                f"url={release.get('html_url', '')}"
            )
            lines.append(line)
    return bounded_text("\n".join(lines), limit=2400)


def _extract_day_hours(visible_text: str, weekday: str) -> str:
    """Extract one weekday's opening hours from a normalized page body."""

    hours_match = re.search(
        r"Opening Hours\s+(?P<hours>.*?)(?:Featured in|More in|What People Say|$)",
        visible_text,
        flags=re.IGNORECASE,
    )
    if not hours_match:
        return ""
    hours_text = hours_match.group("hours")
    day_pattern = re.escape(weekday)
    day_match = re.search(
        rf"{day_pattern}\s+(?P<hours>.*?)(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|$)",
        hours_text,
        flags=re.IGNORECASE,
    )
    if not day_match:
        return ""
    return bounded_text(day_match.group("hours"), limit=160)


def _extract_foodguide_cards(raw_html: str) -> list[dict[str, str]]:
    """Extract candidate restaurant cards from a Food Guide listing page."""

    cards: list[dict[str, str]] = []
    card_pattern = re.compile(
        r'<div data-reviews="(?P<reviews>\d+)" '
        r'data-rating="(?P<rating>[\d.]+)"[^>]*>'
        r'(?P<body>.*?)(?=<div data-reviews="|\Z)',
        flags=re.DOTALL,
    )
    for match in card_pattern.finditer(raw_html):
        body = match.group("body")
        text = _strip_html_fragment(body)
        if "japanese" not in text.lower():
            continue
        href_match = re.search(r'<a href="(?P<href>[^"]+)"', body)
        name_match = re.search(
            r"<h3[^>]*>(?P<name>.*?)</h3>",
            body,
            flags=re.DOTALL,
        )
        if not href_match or not name_match:
            continue
        card = {
            "name": bounded_text(
                _strip_html_fragment(name_match.group("name")),
                limit=120,
            ),
            "url": urljoin(
                "https://foodguide.nz",
                href_match.group("href"),
            ),
            "rating": match.group("rating"),
            "reviews": match.group("reviews"),
            "summary": bounded_text(text, limit=600),
        }
        cards.append(card)
        if len(cards) >= PUBLIC_RESTAURANT_DIRECTORY_MAX_ROWS:
            break
    return cards


def _extract_foodguide_address(raw_html: str) -> str:
    """Extract the visible Google Maps address from a Food Guide detail page."""

    for match in re.finditer(
        r'<a href="https://maps\.google\.com/[^"]+"[^>]*>(?P<text>.*?)</a>',
        raw_html,
        flags=re.DOTALL,
    ):
        text = _strip_html_fragment(match.group("text"))
        if "Auckland" in text:
            return bounded_text(text, limit=180)
    return ""


async def _restaurant_directory_fallback(
    query: str,
    weekday: str,
) -> list[dict[str, str]]:
    """Read public restaurant-directory pages for local dining evidence."""

    if not _looks_like_auckland_japanese_restaurant_query(query):
        return_value: list[dict[str, str]] = []
        return return_value

    try:
        listing_html = await _fetch_public_url_text(
            FOODGUIDE_AUCKLAND_SEARCH_URL,
            PUBLIC_SEARCH_TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as exc:
        return_value = [
            {
                "source": "Food Guide NZ",
                "error": f"restaurant directory fetch failed: {exc}",
            }
        ]
        return return_value

    rows: list[dict[str, str]] = []
    for card in _extract_foodguide_cards(listing_html):
        try:
            detail_html = await _fetch_public_url_text(
                card["url"],
                PUBLIC_SEARCH_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            row = {
                **card,
                "source": "Food Guide NZ",
                "weekday": weekday,
                "weekday_hours": "",
                "address": "",
                "error": f"restaurant detail fetch failed: {exc}",
            }
            rows.append(row)
            continue

        detail_text = _html_to_visible_text(detail_html)
        row = {
            **card,
            "source": "Food Guide NZ",
            "weekday": weekday,
            "weekday_hours": _extract_day_hours(detail_text, weekday),
            "address": _extract_foodguide_address(detail_html),
            "detail_excerpt": bounded_text(
                _focused_excerpt(
                    detail_text,
                    f"{card['name']} Japanese restaurant Auckland Opening Hours {weekday}",
                ),
                limit=1000,
            ),
            "error": "",
        }
        rows.append(row)
    return rows


def _restaurant_directory_summary_text(rows: list[dict[str, str]]) -> str:
    """Render restaurant-directory rows as concise evidence."""

    lines: list[str] = []
    for row in rows:
        if row.get("error"):
            lines.append(f"{row.get('source', '')}: {row['error']}")
            continue
        line = (
            f"{row.get('source', '')}: {row.get('name', '')}; "
            f"rating={row.get('rating', '')} "
            f"reviews={row.get('reviews', '')}; "
            f"{row.get('weekday', '')} hours={row.get('weekday_hours', '')}; "
            f"address={row.get('address', '')}; "
            f"url={row.get('url', '')}"
        )
        lines.append(line)
    return bounded_text("\n".join(lines), limit=2400)


def _compact_catalog_products(
    raw_text: str,
    *,
    base_url: str,
) -> list[dict[str, Any]]:
    """Project a product-suggest JSON response into compact evidence rows."""

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return_value: list[dict[str, Any]] = []
        return return_value

    resources = parsed.get("resources")
    if not isinstance(resources, dict):
        return_value = []
        return return_value
    results = resources.get("results")
    if not isinstance(results, dict):
        return_value = []
        return return_value
    products = results.get("products")
    if not isinstance(products, list):
        return_value = []
        return return_value

    compact_products: list[dict[str, Any]] = []
    for product in products[:PUBLIC_CATALOG_PRODUCTS_PER_TERM]:
        if not isinstance(product, dict):
            continue
        raw_url = str(product.get("url", "")).strip()
        product_url = urljoin(base_url, raw_url)
        compact_product = {
            "title": bounded_text(product.get("title", ""), limit=240),
            "available": product.get("available"),
            "price": bounded_text(product.get("price", ""), limit=80),
            "vendor": bounded_text(product.get("vendor", ""), limit=120),
            "type": bounded_text(product.get("type", ""), limit=160),
            "url": product_url,
        }
        compact_products.append(compact_product)

    return compact_products


def _catalog_summary_text(catalog_rows: list[dict[str, Any]]) -> str:
    """Render product-catalog rows as concise evidence for the verifier."""

    lines: list[str] = []
    for row in catalog_rows:
        if not isinstance(row, dict):
            continue
        products = row.get("products")
        if not isinstance(products, list):
            continue
        for product in products[:2]:
            if not isinstance(product, dict):
                continue
            line = (
                f"{row.get('catalog', '')} query={row.get('query', '')}: "
                f"{product.get('title', '')}; "
                f"available={product.get('available')}; "
                f"price={product.get('price', '')}; "
                f"vendor={product.get('vendor', '')}; "
                f"url={product.get('url', '')}"
            )
            lines.append(line)
    summary = "\n".join(lines)
    return_value = bounded_text(summary, limit=2200)
    return return_value


def _catalog_product_row(
    catalog_row: dict[str, Any],
    product: dict[str, Any],
) -> dict[str, Any]:
    """Project one product suggestion into a hardware evidence row."""

    row = {
        "catalog": bounded_text(catalog_row.get("catalog", ""), limit=120),
        "query": bounded_text(catalog_row.get("query", ""), limit=120),
        "title": bounded_text(product.get("title", ""), limit=220),
        "available": product.get("available"),
        "price": bounded_text(product.get("price", ""), limit=80),
        "vendor": bounded_text(product.get("vendor", ""), limit=120),
        "type": bounded_text(product.get("type", ""), limit=140),
        "url": bounded_text(product.get("url", ""), limit=420),
    }
    return row


def _hardware_catalog_evidence(
    catalog_rows: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Extract hardware-shaped catalog rows without judging sufficiency."""

    ready_systems: list[dict[str, Any]] = []
    graphics_products: list[dict[str, Any]] = []
    component_products: list[dict[str, Any]] = []

    for catalog_row in catalog_rows:
        if not isinstance(catalog_row, dict):
            continue
        products = catalog_row.get("products")
        if not isinstance(products, list):
            continue
        for product in products:
            if not isinstance(product, dict):
                continue
            if product.get("available") is not True:
                continue
            title = str(product.get("title", ""))
            product_type = str(product.get("type", ""))
            combined_text = f"{title} {product_type}".lower()
            row = _catalog_product_row(catalog_row, product)
            if any(marker in combined_text for marker in READY_SYSTEM_MARKERS):
                ready_systems.append(row)
            if any(marker in combined_text for marker in GRAPHICS_PRODUCT_MARKERS):
                graphics_products.append(row)
            if any(marker in combined_text for marker in COMPONENT_PRODUCT_MARKERS):
                component_products.append(row)

    evidence: dict[str, list[dict[str, Any]]] = {}
    if ready_systems:
        evidence["available_ready_systems"] = ready_systems[:4]
    if graphics_products:
        evidence["available_graphics_products"] = graphics_products[:4]
    if component_products:
        evidence["available_component_products"] = component_products[:6]
    return evidence


def _hardware_catalog_summary_text(
    hardware_evidence: dict[str, list[dict[str, Any]]],
) -> str:
    """Render hardware-shaped catalog evidence for LLM handoff."""

    lines: list[str] = []
    for section_name in [
        "available_ready_systems",
        "available_graphics_products",
        "available_component_products",
    ]:
        rows = hardware_evidence.get(section_name, [])
        for row in rows:
            line = (
                f"{section_name}: {row.get('title', '')}; "
                f"available={row.get('available')}; "
                f"price={row.get('price', '')}; "
                f"vendor={row.get('vendor', '')}; "
                f"type={row.get('type', '')}; "
                f"source_query={row.get('query', '')}; "
                f"url={row.get('url', '')}"
            )
            lines.append(line)
    summary = "\n".join(lines)
    return_value = bounded_text(summary, limit=2600)
    return return_value


async def _public_catalog_fallback(query: str) -> list[dict[str, Any]]:
    """Read bounded product-catalog suggestions for product availability tasks."""

    terms = _catalog_search_terms(query)
    if not terms:
        return_value: list[dict[str, Any]] = []
        return return_value

    catalog_rows: list[dict[str, Any]] = []
    for catalog in _PUBLIC_PRODUCT_CATALOGS:
        base_url = catalog["base_url"]
        url_template = catalog["url_template"]
        for term in terms:
            url = url_template.format(query=quote_plus(term))
            try:
                raw_text = await _fetch_public_url_text(
                    url,
                    PUBLIC_CATALOG_TIMEOUT_SECONDS,
                )
            except httpx.HTTPError as exc:
                catalog_rows.append(
                    {
                        "catalog": catalog["name"],
                        "query": term,
                        "url": url,
                        "error": f"catalog read failed: {exc}",
                        "products": [],
                    }
                )
                continue
            products = _compact_catalog_products(raw_text, base_url=base_url)
            catalog_rows.append(
                {
                    "catalog": catalog["name"],
                    "query": term,
                    "url": url,
                    "products": products,
                }
            )

    return catalog_rows


def _rag_context(query: str, state: dict[str, Any]) -> dict[str, Any]:
    """Build the typed debug context required by the existing RAG supervisor."""

    context = {
        "source": "goal_resolver_poc",
        "platform": "debug",
        "platform_channel_id": state["case_id"],
        "channel_type": "direct",
        "platform_message_id": f"goal_resolver_poc:{state['case_id']}",
        "platform_user_id": "goal_resolver_poc_user",
        "global_user_id": "goal_resolver_poc_user",
        "user_name": "goal_resolver_poc_user",
        "local_time_context": state["local_time_context"],
        "current_timestamp_utc": state["local_time_context"][
            "current_utc_datetime"
        ],
        "storage_timestamp_utc": state["local_time_context"][
            "current_utc_datetime"
        ],
        "prompt_message_context": {
            "body_text": query,
            "addressed_to_global_user_ids": [],
            "broadcast": False,
            "mentions": [],
            "attachments": [],
        },
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
    }
    return context


async def execute_tool(
    action: dict[str, Any],
    state: dict[str, Any],
    case: dict[str, Any],
    observation_id: str,
) -> dict[str, Any]:
    """Execute one bounded resolver action.

    Args:
        action: Planner-selected action.
        state: Current resolver state with sandbox and time context.
        case: Case metadata used only for safe fixture scoping.
        observation_id: Stable id assigned by the runner.

    Returns:
        A serializable tool observation.
    """

    tool = action["tool"]
    target_requirement_id = action["target_requirement_id"]
    query = action["query"]
    if tool == "rag_research":
        observation = await _rag_research(
            query,
            state,
            observation_id,
            target_requirement_id,
        )
    elif tool == "web_research":
        observation = await _web_research(
            query,
            state,
            observation_id,
            target_requirement_id,
        )
    elif (
        tool in {"workspace_inspect", "workspace_command"}
        and case["context_hints"].get("fixture") == "incident_logs"
    ):
        observation = _local_artifact_inspect(
            case,
            observation_id,
            target_requirement_id,
        )
        observation["summary"] = (
            f"redirected {tool} to local artifact inspection; "
            f"{observation['summary']}"
        )
        observation["payload"]["redirected_from_tool"] = tool
    elif tool == "workspace_inspect":
        observation = _workspace_inspect(
            state,
            case,
            observation_id,
            target_requirement_id,
        )
    elif tool == "workspace_command":
        observation = _workspace_command(
            state,
            case,
            observation_id,
            target_requirement_id,
        )
    elif tool == "workspace_patch":
        observation = await _workspace_patch(
            query,
            state,
            case,
            observation_id,
            target_requirement_id,
        )
    elif tool == "local_artifact_inspect":
        observation = _local_artifact_inspect(
            case,
            observation_id,
            target_requirement_id,
        )
    elif tool == "self_goal_generate":
        observation = await _self_goal_generate(
            query,
            state,
            observation_id,
            target_requirement_id,
        )
    elif tool == "prepare_action":
        observation = _prepare_action(
            action,
            state,
            observation_id,
            target_requirement_id,
        )
    elif tool == "ask_human":
        observation = _ask_human(
            query,
            observation_id,
            target_requirement_id,
        )
    else:
        observation = _observation(
            observation_id=observation_id,
            tool=tool,
            target_requirement_id=target_requirement_id,
            status="blocked",
            summary=f"Unsupported tool requested: {tool}",
            payload={"query": query},
        )
    return observation


async def _rag_research(
    query: str,
    state: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Use the existing RAG supervisor for internal or mixed evidence."""

    context = _rag_context(query, state)
    result = await call_rag_supervisor(
        query,
        character_name="Kazusa",
        context=context,
    )
    payload = {
        "query": query,
        "answer": bounded_text(result.get("answer", ""), limit=6000),
        "known_facts": result.get("known_facts", []),
        "unknown_slots": result.get("unknown_slots", []),
        "loop_count": result.get("loop_count", 0),
    }
    summary = bounded_text(payload["answer"], limit=500)
    observation = _observation(
        observation_id=observation_id,
        tool="rag_research",
        target_requirement_id=target_requirement_id,
        status="observed",
        summary=summary,
        payload=payload,
    )
    return observation


async def _web_research(
    query: str,
    state: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Use the existing web helper agent for public evidence."""

    context = {
        "source": "goal_resolver_poc",
        "local_time_context": state["local_time_context"],
    }
    raw_search_fallback = ""
    public_search_fallback = {
        "status": "not_used",
        "url": "",
        "text": "",
        "error": "",
    }
    public_search_items = await _public_search_items(query)
    public_page_excerpts = await _public_page_excerpts(
        query,
        public_search_items,
    )
    restaurant_query_context = f"{state['user_input']}\n{query}"
    restaurant_directory_fallback = await _restaurant_directory_fallback(
        restaurant_query_context,
        state["local_time_context"]["current_local_weekday"],
    )
    github_release_fallback = await _github_release_fallback(query)
    has_public_page_evidence = (
        bool(public_search_items["items"]) or bool(public_page_excerpts)
    )
    has_relevant_public_page_evidence = _has_relevant_public_search_items(
        query,
        public_search_items,
    )
    is_model_requirement_query = _looks_like_model_requirement_query(query)
    catalog_query_context = f"{state['user_input']}\n{query}"
    public_catalog_fallback = await _public_catalog_fallback(
        catalog_query_context
    )
    hardware_catalog_evidence = _hardware_catalog_evidence(
        public_catalog_fallback
    )
    has_public_catalog = any(
        row.get("products")
        for row in public_catalog_fallback
        if isinstance(row, dict)
    )
    has_restaurant_directory = any(
        not row.get("error")
        for row in restaurant_directory_fallback
        if isinstance(row, dict)
    )
    has_github_releases = any(
        row.get("releases")
        for row in github_release_fallback
        if isinstance(row, dict)
    )
    has_skip_ready_catalog = has_public_catalog and not is_model_requirement_query
    has_skip_ready_page = (
        has_public_page_evidence and has_relevant_public_page_evidence
    )
    if (
        has_skip_ready_catalog
        or has_skip_ready_page
        or has_restaurant_directory
        or has_github_releases
    ):
        result = {
            "resolved": False,
            "result": "",
            "attempts": 0,
            "knowledge_metadata": {
                "slow_agent_skipped": (
                    "lightweight public evidence was available"
                )
            },
        }
    else:
        result = await WebAgent3().run(query, context, max_attempts=3)

    if (
        not bool(result.get("resolved", False))
        and not has_public_catalog
        and not has_relevant_public_page_evidence
        and not has_restaurant_directory
        and not has_github_releases
    ):
        try:
            raw_search_fallback = await searxng_tools.web_search.ainvoke(
                {"query": query}
            )
        except (McpError, OSError, TimeoutError) as exc:
            raw_search_fallback = f"raw search fallback failed: {exc}"
        public_search_fallback = await _public_search_fallback(query)
    has_public_search = public_search_fallback["status"] == "observed"
    payload = {
        "query": query,
        "resolved": bool(result.get("resolved", False)),
        "result": bounded_text(result.get("result", ""), limit=7000),
        "raw_search_fallback": bounded_text(raw_search_fallback, limit=5000),
        "public_search_items": public_search_items,
        "public_page_excerpts": public_page_excerpts,
        "public_search_fallback": {
            "status": public_search_fallback["status"],
            "url": public_search_fallback["url"],
            "text": bounded_text(
                public_search_fallback["text"],
                limit=PUBLIC_SEARCH_MAX_CHARS,
            ),
            "error": bounded_text(public_search_fallback["error"], limit=600),
        },
        "public_search_fallback_text": bounded_text(
            public_search_fallback["text"],
            limit=PUBLIC_SEARCH_MAX_CHARS,
        ),
        "public_catalog_fallback": public_catalog_fallback,
        "hardware_catalog_evidence": hardware_catalog_evidence,
        "restaurant_directory_fallback": restaurant_directory_fallback,
        "github_release_fallback": github_release_fallback,
        "attempts": result.get("attempts", 0),
        "knowledge_metadata": result.get("knowledge_metadata", {}),
    }
    summary_source = payload["result"]
    if raw_search_fallback:
        summary_source = f"{summary_source}\nRaw search fallback:\n{raw_search_fallback}"
    if public_search_fallback["text"]:
        summary_source = (
            f"{summary_source}\nPublic RSS search fallback:\n"
            f"{public_search_fallback['text']}"
        )
    elif public_search_fallback["error"]:
        summary_source = (
            f"{summary_source}\nPublic RSS search fallback error:\n"
            f"{public_search_fallback['error']}"
        )
    lower_query = query.lower()
    is_technical_model_query = is_model_requirement_query
    catalog_summary = ""
    if has_public_catalog:
        catalog_summary = _catalog_summary_text(public_catalog_fallback)
    hardware_summary = ""
    if hardware_catalog_evidence:
        hardware_summary = _hardware_catalog_summary_text(
            hardware_catalog_evidence
        )
    if has_public_catalog and not is_technical_model_query:
        summary_source = (
            f"Public product catalog evidence:\n{catalog_summary}\n"
            f"{summary_source}"
        )
    if has_restaurant_directory:
        restaurant_summary = _restaurant_directory_summary_text(
            restaurant_directory_fallback
        )
        summary_source = (
            f"Public restaurant directory evidence:\n"
            f"{restaurant_summary}\n{summary_source}"
        )
    if has_github_releases:
        release_summary = _github_release_summary_text(github_release_fallback)
        summary_source = (
            f"Public GitHub release evidence:\n"
            f"{release_summary}\n{summary_source}"
        )
    if public_search_items["items"]:
        search_items_summary = json.dumps(
            public_search_items,
            ensure_ascii=False,
            default=str,
        )
        summary_source = (
            f"{summary_source}\nPublic RSS parsed items:\n"
            f"{search_items_summary}"
        )
    if public_page_excerpts:
        page_excerpt_summary = json.dumps(
            public_page_excerpts,
            ensure_ascii=False,
            default=str,
        )
        summary_source = (
            f"{summary_source}\nPublic page excerpts:\n"
            f"{page_excerpt_summary}"
        )
    if has_public_catalog and is_technical_model_query:
        summary_source = (
            f"{summary_source}\nPublic product catalog evidence:\n"
            f"{catalog_summary}"
        )
    if hardware_summary:
        summary_source = (
            f"Hardware catalog evidence:\n{hardware_summary}\n"
            f"{summary_source}"
        )
    summary = bounded_text(summary_source, limit=1400)
    status = "incomplete"
    if payload["resolved"] or has_public_search or has_public_catalog:
        status = "observed"
    observation = _observation(
        observation_id=observation_id,
        tool="web_research",
        target_requirement_id=target_requirement_id,
        status=status,
        summary=summary,
        payload=payload,
    )
    return observation


def _workspace_inspect(
    state: dict[str, Any],
    case: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Inspect the local code-repair fixture through an isolated sandbox."""

    context_hints = case["context_hints"]
    fixture = context_hints.get("fixture")
    if fixture != "code_repair":
        payload = {"message": "workspace inspection is limited to fixtures"}
        observation = _observation(
            observation_id=observation_id,
            tool="workspace_inspect",
            target_requirement_id=target_requirement_id,
            status="blocked",
            summary="workspace inspection blocked outside approved fixture",
            payload=payload,
        )
        return observation

    sandbox_root = _ensure_code_sandbox(state)
    snapshot = _directory_snapshot(sandbox_root)
    payload = {
        "source_fixture_root": str(CODE_REPAIR_FIXTURE_ROOT),
        "sandbox_root": str(sandbox_root),
        "files": snapshot,
        "validation_command": _sandbox_validation_command(sandbox_root),
    }
    observation = _observation(
        observation_id=observation_id,
        tool="workspace_inspect",
        target_requirement_id=target_requirement_id,
        status="observed",
        summary=f"inspected {len(snapshot)} files in code repair sandbox",
        payload=payload,
    )
    return observation


def _sandbox_validation_command(sandbox_root: Path) -> list[str]:
    """Build the allowlisted Python validation command for a sandbox."""

    python_path = REPO_ROOT / "venv" / "Scripts" / "python.exe"
    if not python_path.exists():
        python_path = REPO_ROOT / "venv" / "Scripts" / "python"
    command = [str(python_path), str(sandbox_root / "run_check.py")]
    return command


def _workspace_command(
    state: dict[str, Any],
    case: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Run the allowlisted code-repair validation command."""

    context_hints = case["context_hints"]
    fixture = context_hints.get("fixture")
    if fixture != "code_repair":
        payload = {"message": "workspace command is limited to code fixture"}
        observation = _observation(
            observation_id=observation_id,
            tool="workspace_command",
            target_requirement_id=target_requirement_id,
            status="blocked",
            summary="workspace command blocked outside approved fixture",
            payload=payload,
        )
        return observation

    sandbox_root = _ensure_code_sandbox(state)
    command = _sandbox_validation_command(sandbox_root)
    completed_process = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=COMMAND_TIMEOUT_SECONDS,
        check=False,
    )
    stdout = bounded_text(completed_process.stdout, limit=3000)
    stderr = bounded_text(completed_process.stderr, limit=3000)
    payload = {
        "command": command,
        "returncode": completed_process.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "sandbox_root": str(sandbox_root),
    }
    if completed_process.returncode == 0:
        status = "passed"
        summary = "validation command passed"
    else:
        status = "failed"
        summary = f"validation command failed with {completed_process.returncode}"
    observation = _observation(
        observation_id=observation_id,
        tool="workspace_command",
        target_requirement_id=target_requirement_id,
        status=status,
        summary=summary,
        payload=payload,
    )
    return observation


async def _workspace_patch(
    query: str,
    state: dict[str, Any],
    case: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Apply an LLM-generated patch inside the code-repair sandbox only."""

    context_hints = case["context_hints"]
    fixture = context_hints.get("fixture")
    if fixture != "code_repair":
        payload = {"message": "workspace patch is limited to code fixture"}
        observation = _observation(
            observation_id=observation_id,
            tool="workspace_patch",
            target_requirement_id=target_requirement_id,
            status="blocked",
            summary="workspace patch blocked outside approved fixture",
            payload=payload,
        )
        return observation

    sandbox_root = _ensure_code_sandbox(state)
    snapshot = _directory_snapshot(sandbox_root)
    command_observations = [
        observation
        for observation in state["tool_history"]
        if observation["tool"] == "workspace_command"
    ]
    patch_request = {
        "query": query,
        "sandbox_root": str(sandbox_root),
        "files": snapshot,
        "recent_command_observations": command_observations[-2:],
    }
    patch = await call_sandbox_patcher(patch_request)
    relative_path_text = patch["file_path"]
    if not relative_path_text or not patch["new_content"]:
        raise ValueError("sandbox patcher returned no file content")
    relative_path = _normalize_sandbox_relative_path(relative_path_text)
    target_path = _resolve_inside(sandbox_root / relative_path, sandbox_root)
    if not target_path.exists():
        raise FileNotFoundError(f"sandbox patch target does not exist: {target_path}")
    if target_path.suffix != ".py":
        raise ValueError(f"sandbox patch target is not a Python file: {target_path}")
    if target_path.name == "run_check.py":
        raise ValueError("sandbox patch target cannot be the validation command")
    target_path.write_text(patch["new_content"], encoding="utf-8")
    payload = {
        "sandbox_root": str(sandbox_root),
        "file_path": str(target_path),
        "reason": patch["reason"],
        "new_content_preview": bounded_text(patch["new_content"], limit=2000),
        "raw_output": patch["raw_output"],
    }
    observation = _observation(
        observation_id=observation_id,
        tool="workspace_patch",
        target_requirement_id=target_requirement_id,
        status="patched",
        summary=f"patched sandbox file {relative_path.as_posix()}: {patch['reason']}",
        payload=payload,
    )
    return observation


def _local_artifact_inspect(
    case: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Inspect approved local incident artifacts."""

    context_hints = case["context_hints"]
    fixture = context_hints.get("fixture")
    if fixture != "incident_logs":
        payload = {"message": "artifact inspection is limited to fixtures"}
        observation = _observation(
            observation_id=observation_id,
            tool="local_artifact_inspect",
            target_requirement_id=target_requirement_id,
            status="blocked",
            summary="artifact inspection blocked outside approved fixture",
            payload=payload,
        )
        return observation
    if not INCIDENT_LOG_FIXTURE_ROOT.exists():
        payload = {"path": str(INCIDENT_LOG_FIXTURE_ROOT)}
        observation = _observation(
            observation_id=observation_id,
            tool="local_artifact_inspect",
            target_requirement_id=target_requirement_id,
            status="missing",
            summary="incident log fixture path is missing",
            payload=payload,
        )
        return observation
    snapshot = _directory_snapshot(INCIDENT_LOG_FIXTURE_ROOT)
    payload = {
        "artifact_root": str(INCIDENT_LOG_FIXTURE_ROOT),
        "files": snapshot,
    }
    observation = _observation(
        observation_id=observation_id,
        tool="local_artifact_inspect",
        target_requirement_id=target_requirement_id,
        status="observed",
        summary=f"inspected {len(snapshot)} incident artifact files",
        payload=payload,
    )
    return observation


async def _self_goal_generate(
    query: str,
    state: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Generate a bounded internal objective candidate set."""

    payload = await call_self_goal_generator(
        {
            "query": query,
            "user_input": state["user_input"],
            "local_time_context": state["local_time_context"],
            "constraints": (
                "目标必须有限、可验证，不能需要对外发送消息、生产副作用、"
                "未批准的 workspace_command 或等待用户回复。"
            ),
        }
    )
    summary = bounded_text(
        (
            f"selected_goal={payload.get('selected_goal', '')}; "
            f"completed_result={payload.get('completed_result', '')}; "
            f"verification_result={payload.get('verification_result', '')}"
        ),
        limit=700,
    )
    observation = _observation(
        observation_id=observation_id,
        tool="self_goal_generate",
        target_requirement_id=target_requirement_id,
        status="observed",
        summary=summary,
        payload=payload,
    )
    return observation


def _prepare_action(
    action: dict[str, Any],
    state: dict[str, Any],
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Create a guarded action candidate without executing it."""

    candidate_id = f"candidate-{len(state['action_candidates']) + 1:03d}"
    candidate = {
        "candidate_id": candidate_id,
        "action_kind": "permissioned_follow_up",
        "summary": bounded_text(action["query"], limit=700),
        "reason": bounded_text(action["reason"], limit=700),
        "expected_effect": (
            "Creates or executes a future reminder/follow-up only after the "
            "user explicitly approves it."
        ),
        "requires_approval": True,
        "execution_status": "not_executed",
        "mapped_action_spec": None,
    }
    state["action_candidates"].append(candidate)
    observation = _observation(
        observation_id=observation_id,
        tool="prepare_action",
        target_requirement_id=target_requirement_id,
        status="pending_approval",
        summary=f"prepared guarded action {candidate_id}",
        payload={"action_candidate": candidate},
    )
    return observation


def _ask_human(
    question: str,
    observation_id: str,
    target_requirement_id: str,
) -> dict[str, Any]:
    """Record a human-input request as a suspended observation."""

    payload = {"question": question}
    observation = _observation(
        observation_id=observation_id,
        tool="ask_human",
        target_requirement_id=target_requirement_id,
        status="needs_human",
        summary=bounded_text(question, limit=500),
        payload=payload,
    )
    return observation
