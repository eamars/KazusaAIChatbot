"""Process-local HTTP(S) URL reader for web_agent3."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.util import find_spec
from html.parser import HTMLParser
from urllib.parse import ParseResult, urlparse

import httpx

from kazusa_ai_chatbot.config import (
    WEB_URL_READ_MAX_BYTES,
    WEB_URL_READ_MAX_CHARS,
    WEB_URL_READ_REDIRECT_LIMIT,
    WEB_URL_READ_TIMEOUT_SECONDS,
    WEB_URL_READER_ACCEPT_LANGUAGE,
    WEB_URL_READER_USER_AGENT,
)

_ACCEPT_HEADER = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,image/apng,*/*;q=0.8,"
    "application/signed-exchange;v=b3;q=0.7"
)
_READ_ERROR_CHAR_LIMIT = 800
_BINARY_SAMPLE_SIZE = 1024
_BINARY_CONTROL_RATIO = 0.30
_CHALLENGE_BODY_SAMPLE_SIZE = 65536
_CHALLENGE_STATUS_CODES = {403, 429, 503}
_BROTLI_AVAILABLE = (
    find_spec("brotli") is not None
    or find_spec("brotlicffi") is not None
)
_ZSTD_AVAILABLE = find_spec("zstandard") is not None
_COOKIE_JAR = httpx.Cookies()
_COOKIE_LOCK = asyncio.Lock()
_IGNORED_HTML_TAGS = {
    "script",
    "style",
    "noscript",
    "template",
    "svg",
}
_IGNORED_VOID_HTML_TAGS = {
    "meta",
    "link",
}
_PARAGRAPH_HTML_TAGS = {"p", "li", "td", "th", "a", "button"}
_PROVIDER_HEADER_MARKERS = {
    "cloudflare": (
        "__cf_bm",
        "cf-chl",
        "cf-ray",
        "cloudflare",
    ),
    "datadome": (
        "datadome",
        "x-datadome",
    ),
    "akamai": (
        "ak_bmsc",
        "akamai",
        "bm_sz",
    ),
    "perimeterx": (
        "_px",
        "perimeterx",
        "px-captcha",
    ),
}
_PROVIDER_BODY_MARKERS = {
    "cloudflare": (
        "cf-browser-verification",
        "cf-chl",
        "challenge-platform",
        "checking your browser",
        "just a moment...",
    ),
    "datadome": (
        "datadome",
        "dd_captcha",
    ),
    "akamai": (
        "akamai bot manager",
        "akamai ghost",
    ),
    "perimeterx": (
        "perimeterx",
        "px-captcha",
    ),
}
_GENERIC_CHALLENGE_BODY_MARKERS = (
    "are you a human",
    "bot detection",
    "captcha",
    "complete the security check",
    "verify you are human",
)


@dataclass
class _HtmlBlock:
    """One extracted visible text block from an HTML document."""

    kind: str
    text: str
    level: int = 0


class _ReadableHTMLParser(HTMLParser):
    """Extract readable headings and body text from HTML."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[_HtmlBlock] = []
        self.headings: list[_HtmlBlock] = []
        self._ignored_depth = 0
        self._capture_kind = ""
        self._capture_level = 0
        self._capture_end_tags: set[str] = set()
        self._capture_parts: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        """Start capturing visible text for supported HTML tags."""
        del attrs

        normalized_tag = tag.lower()
        if normalized_tag in _IGNORED_VOID_HTML_TAGS:
            return

        if normalized_tag in _IGNORED_HTML_TAGS:
            self._ignored_depth += 1
            return

        if self._ignored_depth:
            return

        if normalized_tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._start_capture(
                kind="heading",
                level=int(normalized_tag[1]),
                end_tags={normalized_tag},
            )
            return

        if normalized_tag == "title":
            self._start_capture(
                kind="title",
                level=0,
                end_tags={"title"},
            )
            return

        if normalized_tag in _PARAGRAPH_HTML_TAGS:
            self._start_capture(
                kind="paragraph",
                level=0,
                end_tags={normalized_tag},
            )
            return

        if normalized_tag == "br" and self._capture_kind:
            self._capture_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        """Finish a supported capture or ignored HTML section."""
        normalized_tag = tag.lower()
        if normalized_tag in _IGNORED_HTML_TAGS and self._ignored_depth:
            self._ignored_depth -= 1
            return

        if self._ignored_depth:
            return

        if normalized_tag in self._capture_end_tags:
            self._finish_capture()

    def handle_data(self, data: str) -> None:
        """Collect text data while a supported tag is active."""
        if self._ignored_depth or not self._capture_kind:
            return

        self._capture_parts.append(data)

    def close(self) -> None:
        """Flush any still-open text capture at end of document."""
        super().close()
        if self._capture_kind:
            self._finish_capture()

    def _start_capture(
        self,
        *,
        kind: str,
        level: int,
        end_tags: set[str],
    ) -> None:
        """Start a new supported text capture."""
        if self._capture_kind:
            self._finish_capture()

        self._capture_kind = kind
        self._capture_level = level
        self._capture_end_tags = set(end_tags)
        self._capture_parts = []

    def _finish_capture(self) -> None:
        """Store the active capture as a normalized readable block."""
        raw_text = " ".join(self._capture_parts)
        normalized_text = " ".join(raw_text.split()).strip()
        if normalized_text:
            block = _HtmlBlock(
                kind=self._capture_kind,
                text=normalized_text,
                level=self._capture_level,
            )
            self.blocks.append(block)
            if block.kind == "heading":
                self.headings.append(block)

        self._capture_kind = ""
        self._capture_level = 0
        self._capture_end_tags = set()
        self._capture_parts = []


def _bounded_error(message: str) -> str:
    """Return a bounded error observation for prompt-facing tool output."""

    if len(message) <= _READ_ERROR_CHAR_LIMIT:
        return message

    clipped_message = message[:_READ_ERROR_CHAR_LIMIT].rstrip()
    return_value = f"{clipped_message}..."
    return return_value


def _error_message(prefix: str, exc: BaseException) -> str:
    """Build a bounded URL-reader error with concrete exception text."""

    message = f"Error: {prefix}: {exc}"
    bounded_message = _bounded_error(message)
    return bounded_message


def _accept_encoding_header() -> str:
    """Advertise only encodings the local HTTP stack can decode."""

    encodings = ["gzip", "deflate"]
    if _BROTLI_AVAILABLE:
        encodings.append("br")

    if _ZSTD_AVAILABLE:
        encodings.append("zstd")

    header = ", ".join(encodings)
    return header


def _reader_headers(parsed_url: ParseResult) -> dict[str, str]:
    """Build browser-compatible request headers for URL reads."""

    referer_netloc = parsed_url.netloc.rsplit("@", maxsplit=1)[-1]
    referer = f"{parsed_url.scheme}://{referer_netloc}/"
    headers = {
        "User-Agent": WEB_URL_READER_USER_AGENT,
        "Accept": _ACCEPT_HEADER,
        "Accept-Language": WEB_URL_READER_ACCEPT_LANGUAGE,
        "Accept-Encoding": _accept_encoding_header(),
        "DNT": "1",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "Referer": referer,
    }
    return headers


async def _cookie_snapshot() -> httpx.Cookies:
    """Copy the process-memory cookie jar for a single URL read."""

    cookies = httpx.Cookies()
    async with _COOKIE_LOCK:
        cookies.update(_COOKIE_JAR)
    return cookies


async def _store_response_cookies(cookies: httpx.Cookies) -> None:
    """Store response cookies for later URL reads in this process."""

    async with _COOKIE_LOCK:
        _COOKIE_JAR.update(cookies)


def _contains_marker(text: str, markers: tuple[str, ...]) -> bool:
    """Return whether text contains any known anti-bot marker."""

    for marker in markers:
        if marker in text:
            return_value = True
            return return_value

    return_value = False
    return return_value


def _headers_to_text(headers: Mapping[str, str]) -> str:
    """Flatten response headers for anti-bot marker inspection."""

    header_parts = []
    for name, value in headers.items():
        header_parts.append(f"{name}: {value}")

    header_text = "\n".join(header_parts).lower()
    return header_text


def _detect_anti_bot_challenge(
    response: httpx.Response,
    content: bytes,
) -> str:
    """Identify common HTTP challenge pages before returning a generic error."""

    status_code = response.status_code
    header_text = _headers_to_text(response.headers)
    sample = content[:_CHALLENGE_BODY_SAMPLE_SIZE]
    body_text = _decode_utf8(sample).lower()

    if status_code not in _CHALLENGE_STATUS_CODES:
        return_value = ""
        return return_value

    for provider, markers in _PROVIDER_BODY_MARKERS.items():
        if _contains_marker(body_text, markers):
            return_value = provider
            return return_value

    for provider, markers in _PROVIDER_HEADER_MARKERS.items():
        if _contains_marker(header_text, markers):
            return_value = provider
            return return_value

    if _contains_marker(body_text, _GENERIC_CHALLENGE_BODY_MARKERS):
        return_value = "unknown"
        return return_value

    return_value = ""
    return return_value


def _anti_bot_challenge_error(provider: str, status_code: int) -> str:
    """Build a bounded prompt-facing anti-bot challenge error."""

    message = (
        "Error: URL read blocked by anti-bot challenge: "
        f"{provider} (HTTP {status_code})"
    )
    bounded_message = _bounded_error(message)
    return bounded_message


def _content_type_from_response(response: httpx.Response) -> str:
    """Return the normalized Content-Type header from an HTTP response."""

    content_type = response.headers.get("content-type", "").lower()
    return content_type


def _looks_like_html(content: bytes) -> bool:
    """Check whether untyped content begins with an HTML document marker."""

    prefix = content[:_BINARY_SAMPLE_SIZE].lstrip().lower()
    looks_like_html = prefix.startswith(b"<!doctype html") or prefix.startswith(
        b"<html"
    )
    return looks_like_html


def _is_html_content(content_type: str, content: bytes) -> bool:
    """Decide whether the response should be parsed as HTML."""

    is_html_type = (
        "text/html" in content_type
        or "application/xhtml+xml" in content_type
    )
    if is_html_type:
        return_value = True
        return return_value

    if not content_type and _looks_like_html(content):
        return_value = True
        return return_value

    return_value = False
    return return_value


def _is_textual_content_type(content_type: str) -> bool:
    """Decide whether a non-HTML response is textual."""

    if not content_type:
        return_value = True
        return return_value

    textual_type = (
        content_type.startswith("text/")
        or "application/json" in content_type
        or "application/xml" in content_type
        or "application/javascript" in content_type
        or "+json" in content_type
        or "+xml" in content_type
    )
    return textual_type


def _looks_binary(content: bytes) -> bool:
    """Detect obvious binary responses before text decoding."""

    if not content:
        return_value = False
        return return_value

    if b"\x00" in content:
        return_value = True
        return return_value

    sample = content[:_BINARY_SAMPLE_SIZE]
    control_count = 0
    for byte_value in sample:
        is_allowed_control = byte_value in (9, 10, 13)
        is_control = byte_value < 32 and not is_allowed_control
        if is_control:
            control_count += 1

    control_ratio = control_count / len(sample)
    return_value = control_ratio > _BINARY_CONTROL_RATIO
    return return_value


def _decode_utf8(content: bytes) -> str:
    """Decode response bytes as UTF-8 with replacement for invalid bytes."""

    text = content.decode("utf-8", errors="replace")
    return text


def _extract_html(content: bytes) -> _ReadableHTMLParser:
    """Parse HTML response bytes into readable text blocks and headings."""

    html_text = _decode_utf8(content)
    parser = _ReadableHTMLParser()
    parser.feed(html_text)
    parser.close()
    return parser


def _blocks_to_text(blocks: list[_HtmlBlock]) -> str:
    """Render extracted text blocks into readable plain text."""

    lines = [block.text for block in blocks if block.text]
    text = "\n".join(lines).strip()
    return text


def _select_section(
    blocks: list[_HtmlBlock],
    section: str,
) -> list[_HtmlBlock] | str:
    """Select content under the first matching heading."""

    section_query = section.strip().lower()
    if not section_query:
        return blocks

    start_index = -1
    start_level = 0
    for index, block in enumerate(blocks):
        if block.kind != "heading":
            continue
        if section_query in block.text.lower():
            start_index = index
            start_level = block.level
            break

    if start_index < 0:
        return_value = f"Error: section not found: {section}"
        return return_value

    selected_blocks: list[_HtmlBlock] = []
    for block in blocks[start_index + 1:]:
        ends_section = block.kind == "heading" and block.level <= start_level
        if ends_section:
            break
        selected_blocks.append(block)

    return selected_blocks


def _parse_positive_int(text: str) -> int | None:
    """Parse a positive one-based integer from user tool arguments."""

    try:
        value = int(text)
    except ValueError:
        return_value = None
        return return_value

    if value < 1:
        return_value = None
        return return_value

    return value


def _parse_paragraph_range(paragraph_range: str) -> tuple[int, int | None] | None:
    """Parse a one-based paragraph range of the forms N, N-M, or N-."""

    range_text = paragraph_range.strip()
    if "-" not in range_text:
        start = _parse_positive_int(range_text)
        if start is None:
            return_value = None
            return return_value
        return_value = (start, start)
        return return_value

    parts = range_text.split("-", maxsplit=1)
    start_text = parts[0].strip()
    end_text = parts[1].strip()
    start = _parse_positive_int(start_text)
    if start is None:
        return_value = None
        return return_value

    if not end_text:
        return_value = (start, None)
        return return_value

    end = _parse_positive_int(end_text)
    if end is None or end < start:
        return_value = None
        return return_value

    return_value = (start, end)
    return return_value


def _select_paragraphs(
    blocks: list[_HtmlBlock],
    paragraph_range: str,
) -> list[_HtmlBlock] | str:
    """Apply one-based paragraph selection to extracted text blocks."""

    range_text = paragraph_range.strip()
    if not range_text:
        return blocks

    parsed_range = _parse_paragraph_range(range_text)
    if parsed_range is None:
        return_value = f"Error: invalid paragraphRange: {paragraph_range}"
        return return_value

    start, end = parsed_range
    paragraphs = [block for block in blocks if block.kind == "paragraph"]
    start_index = start - 1
    if end is None:
        selected_blocks = paragraphs[start_index:]
    else:
        selected_blocks = paragraphs[start_index:end]

    return selected_blocks


def _plain_text_blocks(text: str) -> list[_HtmlBlock]:
    """Build paragraph blocks from non-HTML text for range selection."""

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]
    blocks = [
        _HtmlBlock(kind="paragraph", text=paragraph)
        for paragraph in paragraphs
    ]
    return blocks


def _bounded_slice(text: str, start_char: int, max_length: int) -> str:
    """Apply configured character limits after extraction and filtering."""

    if start_char < 0:
        start = 0
    else:
        start = start_char

    if max_length <= 0:
        length = WEB_URL_READ_MAX_CHARS
    elif max_length > WEB_URL_READ_MAX_CHARS:
        length = WEB_URL_READ_MAX_CHARS
    else:
        length = max_length

    sliced_text = text[start:start + length]
    return sliced_text


def _finalize_text(text: str, start_char: int, max_length: int) -> str:
    """Return sliced text or a bounded empty-content error."""

    normalized_text = text.strip()
    if not normalized_text:
        return_value = "Error: empty content"
        return return_value

    sliced_text = _bounded_slice(normalized_text, start_char, max_length)
    if not sliced_text:
        return_value = "Error: empty content"
        return return_value

    return sliced_text


def _process_html_content(
    content: bytes,
    *,
    start_char: int,
    max_length: int,
    section: str,
    paragraph_range: str,
    read_headings: bool,
) -> str:
    """Extract and filter readable text from an HTML response."""

    parser = _extract_html(content)
    if read_headings:
        heading_lines = [heading.text for heading in parser.headings]
        if not heading_lines:
            return_value = "Error: no headings found"
            return return_value
        headings_text = "\n".join(heading_lines)
        result = _finalize_text(headings_text, start_char, max_length)
        return result

    selected_blocks = _select_section(parser.blocks, section)
    if isinstance(selected_blocks, str):
        return selected_blocks

    paragraph_blocks = _select_paragraphs(selected_blocks, paragraph_range)
    if isinstance(paragraph_blocks, str):
        return paragraph_blocks

    extracted_text = _blocks_to_text(paragraph_blocks)
    result = _finalize_text(extracted_text, start_char, max_length)
    return result


def _process_text_content(
    content: bytes,
    *,
    start_char: int,
    max_length: int,
    section: str,
    paragraph_range: str,
    read_headings: bool,
) -> str:
    """Filter and slice non-HTML textual response content."""

    if read_headings:
        return_value = "Error: no headings found"
        return return_value

    if section.strip():
        return_value = f"Error: section not found: {section}"
        return return_value

    text = _decode_utf8(content)
    if paragraph_range.strip():
        blocks = _plain_text_blocks(text)
        paragraph_blocks = _select_paragraphs(blocks, paragraph_range)
        if isinstance(paragraph_blocks, str):
            return paragraph_blocks
        text = _blocks_to_text(paragraph_blocks)

    result = _finalize_text(text, start_char, max_length)
    return result


async def _fetch_url_content(
    url: str,
    headers: dict[str, str],
) -> tuple[bytes, str] | str:
    """Fetch URL bytes while enforcing the configured streaming byte cap.

    Args:
        url: HTTP(S) URL to fetch.
        headers: Browser-compatible request headers.

    Returns:
        A tuple of response bytes and content type, or a bounded error string.
    """
    content_parts: list[bytes] = []
    content_type = ""
    total_bytes = 0
    cookies = await _cookie_snapshot()
    try:
        async with httpx.AsyncClient(
            timeout=WEB_URL_READ_TIMEOUT_SECONDS,
            follow_redirects=True,
            max_redirects=WEB_URL_READ_REDIRECT_LIMIT,
            cookies=cookies,
        ) as client:
            async with client.stream("GET", url, headers=headers) as response:
                content_type = _content_type_from_response(response)
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    chunk_size = len(chunk)
                    if total_bytes + chunk_size > WEB_URL_READ_MAX_BYTES:
                        message = (
                            "Error: response too large: exceeds "
                            f"WEB_URL_READ_MAX_BYTES={WEB_URL_READ_MAX_BYTES}"
                        )
                        return_value = _bounded_error(message)
                        return return_value
                    content_parts.append(chunk)
                    total_bytes += chunk_size
                await _store_response_cookies(response.cookies)
                client_cookies = getattr(client, "cookies", None)
                if client_cookies is not None:
                    await _store_response_cookies(client_cookies)
                content = b"".join(content_parts)
                provider = _detect_anti_bot_challenge(response, content)
                if provider:
                    return_value = _anti_bot_challenge_error(
                        provider,
                        response.status_code,
                    )
                    return return_value
                response.raise_for_status()
    except httpx.InvalidURL as exc:
        return_value = _error_message("invalid URL", exc)
        return return_value
    except httpx.TimeoutException as exc:
        return_value = _error_message("URL read timed out", exc)
        return return_value
    except httpx.HTTPStatusError as exc:
        return_value = _error_message("URL read HTTP error", exc)
        return return_value
    except httpx.RequestError as exc:
        return_value = _error_message("URL read request failed", exc)
        return return_value

    content = b"".join(content_parts)
    return_value = (content, content_type)
    return return_value


async def web_url_read(
    url: str,
    startChar: int = 0,
    maxLength: int = 10000,
    section: str = "",
    paragraphRange: str = "",
    readHeadings: bool = False,
) -> str:
    """Read HTTP(S) URL content with local extraction and bounded output.

    Args:
        url: HTTP(S) URL to read.
        startChar: Zero-based character offset applied after extraction.
        maxLength: Maximum characters to return, capped by config.
        section: Optional case-insensitive heading query for HTML sections.
        paragraphRange: Optional one-based paragraph range: N, N-M, or N-.
        readHeadings: Whether to return HTML headings instead of body text.

    Returns:
        Readable bounded text, or a bounded error observation.
    """
    try:
        parsed_url = urlparse(url)
    except ValueError as exc:
        return_value = _error_message("invalid URL", exc)
        return return_value

    scheme = parsed_url.scheme.lower()
    if scheme not in ("http", "https"):
        return_value = f"Error: unsupported URL scheme: {scheme}"
        return return_value

    if not parsed_url.netloc:
        return_value = "Error: invalid URL: missing network location"
        return return_value

    headers = _reader_headers(parsed_url)
    fetched_content = await _fetch_url_content(url, headers)
    if isinstance(fetched_content, str):
        return_value = fetched_content
        return return_value

    content, content_type = fetched_content

    if not content:
        return_value = "Error: empty content"
        return return_value

    if _is_html_content(content_type, content):
        result = _process_html_content(
            content,
            start_char=startChar,
            max_length=maxLength,
            section=section,
            paragraph_range=paragraphRange,
            read_headings=readHeadings,
        )
        return result

    if not _is_textual_content_type(content_type) or _looks_binary(content):
        return_value = "Error: unsupported binary content"
        return return_value

    result = _process_text_content(
        content,
        start_char=startChar,
        max_length=maxLength,
        section=section,
        paragraph_range=paragraphRange,
        read_headings=readHeadings,
    )
    return result
