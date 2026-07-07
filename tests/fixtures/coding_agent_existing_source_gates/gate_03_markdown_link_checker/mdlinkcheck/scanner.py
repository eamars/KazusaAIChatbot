"""Scan Markdown files for local link problems."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from mdlinkcheck.anchors import collect_anchors


LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


@dataclass(frozen=True)
class MarkdownLink:
    text: str
    target: str


@dataclass(frozen=True)
class LinkProblem:
    path: str
    message: str


def find_markdown_links(markdown: str) -> list[MarkdownLink]:
    """Find Markdown inline links in a document."""

    links: list[MarkdownLink] = []
    for match in LINK_RE.finditer(markdown):
        links.append(MarkdownLink(text=match.group(1), target=match.group(2)))

    return links


def check_file(path: Path, root: Path) -> list[LinkProblem]:
    """Check one Markdown file for broken local Markdown links."""

    markdown = path.read_text(encoding="utf-8")
    current_anchors = collect_anchors(markdown)
    problems: list[LinkProblem] = []

    for slug, count in current_anchors.items():
        if count > 1:
            problems.append(LinkProblem(str(path), f"duplicate anchor: {slug}"))

    for link in find_markdown_links(markdown):
        target = link.target
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        target_path_text, _, anchor = target.partition("#")
        target_path = path if not target_path_text else path.parent / target_path_text
        if not target_path.exists():
            problems.append(LinkProblem(str(path), f"missing target: {target}"))
            continue
        if anchor:
            target_anchors = collect_anchors(target_path.read_text(encoding="utf-8"))
            if anchor not in target_anchors:
                problems.append(LinkProblem(str(path), f"missing anchor: {target}"))

    return problems
