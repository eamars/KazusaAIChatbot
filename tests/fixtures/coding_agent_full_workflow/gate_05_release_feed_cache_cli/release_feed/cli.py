"""Command-line interface for release feed rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from release_feed.feed import render_titles


def main(argv: list[str] | None = None) -> int:
    """Print release titles from a JSON fixture file."""

    parser = argparse.ArgumentParser(prog="release-feed")
    parser.add_argument("path", type=Path)
    parser.add_argument("--include-drafts", action="store_true")
    args = parser.parse_args(argv)

    releases = json.loads(args.path.read_text(encoding="utf-8"))
    for title in render_titles(releases):
        print(title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
