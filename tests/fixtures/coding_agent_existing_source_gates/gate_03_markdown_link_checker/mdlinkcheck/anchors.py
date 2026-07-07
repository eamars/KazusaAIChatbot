"""Collect Markdown heading anchors."""

from __future__ import annotations

import re


HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def slugify_heading(text: str) -> str:
    """Return a simple GitHub-like slug for one heading."""

    lowered = text.strip().lower()
    slug = re.sub(r"[^a-z0-9\s-]", "", lowered)
    slug = re.sub(r"\s+", "-", slug)
    slug = slug.strip("-")
    return slug


def collect_anchors(markdown: str) -> dict[str, int]:
    """Collect heading anchors and occurrence counts from Markdown text."""

    anchors: dict[str, int] = {}
    for line in markdown.splitlines():
        match = HEADING_RE.match(line)
        if match is None:
            continue
        slug = slugify_heading(match.group(2))
        anchors[slug] = anchors.get(slug, 0) + 1

    return anchors
