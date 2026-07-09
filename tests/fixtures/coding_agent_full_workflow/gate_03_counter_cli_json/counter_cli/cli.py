"""Line-counting command-line interface."""

from __future__ import annotations

import argparse
from pathlib import Path


def count_lines(path: Path) -> int:
    """Count newline-separated rows in a UTF-8 text file."""

    text = path.read_text(encoding="utf-8")
    if not text:
        return 0
    count = len(text.splitlines())
    return count


def render_text(path: Path, line_count: int) -> str:
    """Render the plain text counter summary."""

    return_value = f"{path.name}: {line_count} lines"
    return return_value


def main(argv: list[str] | None = None) -> int:
    """Run the line counter CLI."""

    parser = argparse.ArgumentParser(prog="counter-cli")
    parser.add_argument("path", type=Path)
    args = parser.parse_args(argv)

    line_count = count_lines(args.path)
    print(render_text(args.path, line_count))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
