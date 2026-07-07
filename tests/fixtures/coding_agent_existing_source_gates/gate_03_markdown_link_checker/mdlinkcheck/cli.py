"""Command-line interface for local Markdown link checks."""

from __future__ import annotations

import argparse
from pathlib import Path

from mdlinkcheck.scanner import check_file


def main(argv: list[str] | None = None) -> int:
    """Run link checks for Markdown files under a directory."""

    parser = argparse.ArgumentParser(description="Check local Markdown links.")
    parser.add_argument("root", type=Path)
    args = parser.parse_args(argv)

    problems = []
    for path in sorted(args.root.rglob("*.md")):
        problems.extend(check_file(path, args.root))

    for problem in problems:
        print(f"{problem.path}: {problem.message}")

    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
