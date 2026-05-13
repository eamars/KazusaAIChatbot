"""Run a local self-cognition dry-run case file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.self_cognition.projection import validate_case_name
from kazusa_ai_chatbot.self_cognition.runner import run_self_cognition_case


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run one local self-cognition dry-run case.",
    )
    parser.add_argument(
        "--case-file",
        type=Path,
        required=True,
        help="Path to a JSON self-cognition case file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local artifact output directory.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the dry-run CLI and return a process exit code."""

    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        case = _load_case_file(args.case_file)
        validate_case_name(case)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"self-cognition dry-run rejected case file: {exc}", file=sys.stderr)
        return_value = 1
        return return_value

    paths = run_self_cognition_case(case, args.output_dir)
    print(f"wrote {len(paths)} self-cognition artifact(s) to {args.output_dir}")
    return_value = 0
    return return_value


def _load_case_file(case_file: Path) -> dict[str, Any]:
    """Load and structurally validate a JSON case file.

    Args:
        case_file: Path to the external JSON case file.

    Returns:
        Case dictionary.

    Raises:
        FileNotFoundError: If the case file does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level JSON value is not an object.
    """

    content = case_file.read_text(encoding="utf-8")
    data = json.loads(content)
    if not isinstance(data, dict):
        raise ValueError("self-cognition case file must contain a JSON object")
    return data


if __name__ == "__main__":
    sys.exit(main())
