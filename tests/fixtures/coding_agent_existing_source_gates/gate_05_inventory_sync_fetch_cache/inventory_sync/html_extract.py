"""Extract page metadata from simple HTML."""

from __future__ import annotations

from html.parser import HTMLParser


class _MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._active_tag: str | None = None
        self.title = ""
        self.h1 = ""

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"title", "h1"}:
            self._active_tag = tag

    def handle_endtag(self, tag: str) -> None:
        if tag == self._active_tag:
            self._active_tag = None

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text:
            return
        if self._active_tag == "title" and not self.title:
            self.title = text
        if self._active_tag == "h1" and not self.h1:
            self.h1 = text


def extract_page_metadata(html: str) -> dict[str, str]:
    """Extract title and first h1 values from HTML."""

    parser = _MetadataParser()
    parser.feed(html)
    metadata = {"title": parser.title, "h1": parser.h1}
    return metadata
