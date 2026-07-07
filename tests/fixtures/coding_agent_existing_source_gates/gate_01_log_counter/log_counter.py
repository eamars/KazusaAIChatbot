"""Count severity-prefixed log entries from a plain text file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SEVERITIES = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


def count_severities(path: Path) -> tuple[dict[str, int], int]:
    """Count known severity prefixes and malformed lines in a log file."""

    counts = {severity: 0 for severity in SEVERITIES}
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split(" ", 1)
        if len(parts) != 2 or parts[0] not in counts:
            skipped += 1
            continue
        counts[parts[0]] += 1

    return counts, skipped


def format_summary(counts: dict[str, int], skipped: int) -> str:
    """Render a terminal summary for severity counts."""

    lines = [f"{severity}: {counts[severity]}" for severity in SEVERITIES]
    lines.append(f"skipped: {skipped}")
    summary = "\n".join(lines)
    return summary


def main(argv: list[str] | None = None) -> int:
    """Run the command-line log counter."""

    parser = argparse.ArgumentParser(description="Count log severities.")
    parser.add_argument("log_path", type=Path)
    args = parser.parse_args(argv)

    if not args.log_path.is_file():
        print(f"missing input file: {args.log_path}", file=sys.stderr)
        return 2

    counts, skipped = count_severities(args.log_path)
    print(format_summary(counts, skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
