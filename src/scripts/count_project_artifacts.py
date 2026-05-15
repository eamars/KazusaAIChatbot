"""CLI helper to count project source, test, and documentation artifacts.

Typical use:
    python -m scripts.count_project_artifacts
    python -m scripts.count_project_artifacts --json
    count-project-artifacts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

CATEGORY_ORDER = ("production_code", "unit_test_code", "documents")
DOCUMENT_FILENAMES = frozenset({"LICENSE", "NOTICE", "COPYING"})
DOCUMENT_SUFFIXES = frozenset({".adoc", ".md", ".rst", ".txt"})
IGNORED_DIR_NAMES = frozenset({
    ".agents",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
    "node_modules",
    "test_artifacts",
    "venv",
})

ArtifactReport = dict[str, dict[str, int]]


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when the active stream supports it."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for project artifact counts.

    Returns:
        Configured parser for the read-only counting command.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Count production code, unit test code, and documentation files."
        ),
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Project root to scan. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the artifact report as JSON.",
    )
    return parser


def _empty_report() -> ArtifactReport:
    """Build an empty report with stable category ordering.

    Returns:
        Artifact totals initialized to zero for every public category.
    """

    report = {
        category: {"files": 0, "lines": 0}
        for category in CATEGORY_ORDER
    }
    return report


def _is_ignored(relative_path: Path) -> bool:
    """Return whether a repository-relative path is generated or local-only.

    Args:
        relative_path: Path relative to the scan root.

    Returns:
        True when any path segment belongs to an ignored directory.
    """

    ignored = any(part in IGNORED_DIR_NAMES for part in relative_path.parts)
    return ignored


def _category_for(relative_path: Path) -> str | None:
    """Classify a repository-relative file path into a counted category.

    Args:
        relative_path: File path relative to the scan root.

    Returns:
        Category key for counted files, otherwise ``None``.
    """

    parts = relative_path.parts
    suffix = relative_path.suffix.lower()
    filename = relative_path.name

    category = None
    if suffix == ".py":
        if parts and parts[0] == "tests":
            category = "unit_test_code"
        elif parts and parts[0] == "src":
            category = "production_code"
    elif suffix in DOCUMENT_SUFFIXES or filename in DOCUMENT_FILENAMES:
        category = "documents"

    return category


def _line_count(path: Path) -> int:
    """Count physical text lines in a project artifact.

    Args:
        path: UTF-8 text file to inspect.

    Returns:
        Number of physical lines in the file.
    """

    text = path.read_text(encoding="utf-8")
    count = len(text.splitlines())
    return count


def build_artifact_report(root: Path) -> ArtifactReport:
    """Count production code, unit test code, and documents under a root.

    Args:
        root: Project root directory to scan.

    Returns:
        Mapping from category name to file and line totals.
    """

    report = _empty_report()
    resolved_root = root.resolve()
    for path in sorted(resolved_root.rglob("*")):
        if not path.is_file():
            continue

        relative_path = path.relative_to(resolved_root)
        if _is_ignored(relative_path):
            continue

        category = _category_for(relative_path)
        if category is None:
            continue

        report[category]["files"] += 1
        report[category]["lines"] += _line_count(path)

    return report


def format_artifact_report(report: ArtifactReport) -> str:
    """Format an artifact report as stable terminal output.

    Args:
        report: Mapping returned by ``build_artifact_report``.

    Returns:
        Human-readable table with category, file, and line counts.
    """

    rows = [("category", "files", "lines")]
    for category in CATEGORY_ORDER:
        totals = report[category]
        row = (category, str(totals["files"]), str(totals["lines"]))
        rows.append(row)

    widths = [
        max(len(row[column_index]) for row in rows)
        for column_index in range(3)
    ]
    lines = []
    for row_index, row in enumerate(rows):
        rendered_row = (
            f"{row[0]:<{widths[0]}}  "
            f"{row[1]:>{widths[1]}}  "
            f"{row[2]:>{widths[2]}}"
        )
        lines.append(rendered_row)
        if row_index == 0:
            separator = (
                f"{'-' * widths[0]}  "
                f"{'-' * widths[1]}  "
                f"{'-' * widths[2]}"
            )
            lines.append(separator)

    rendered = "\n".join(lines)
    return rendered


def main(argv: Sequence[str] | None = None) -> int:
    """Run the project artifact counting command.

    Args:
        argv: Optional command-line arguments excluding the program name.

    Returns:
        Process exit code.
    """

    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = build_artifact_report(Path(args.root))
    if args.json:
        output = json.dumps(report, indent=2, sort_keys=True)
    else:
        output = format_artifact_report(report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
