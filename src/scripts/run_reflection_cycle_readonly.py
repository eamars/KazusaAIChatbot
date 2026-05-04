"""Run the read-only reflection-cycle evaluation.

Typical use:
    python -m scripts.run_reflection_cycle_readonly --lookback-hours 24
    python -m scripts.run_reflection_cycle_readonly --lookback-hours 24 --real-llm
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from pathlib import Path

from scripts._db_export import configure_logging, configure_stdout, load_project_env

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.reflection_cycle import run_readonly_reflection_evaluation


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Run read-only reflection-cycle evaluation over recent conversation history."
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Requested message evaluation window.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("test_artifacts") / "reflection_cycle_readonly",
        help="Local artifact directory.",
    )
    parser.add_argument(
        "--real-llm",
        action="store_true",
        help="Call the configured consolidation LLM instead of writing prompt-only artifacts.",
    )
    parser.add_argument(
        "--now",
        help="Optional ISO timestamp for deterministic evaluation windows.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project logs.",
    )
    return parser


async def main() -> None:
    """Run the read-only reflection-cycle CLI."""

    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    now = _parse_optional_datetime(args.now)
    try:
        result = await run_readonly_reflection_evaluation(
            lookback_hours=args.lookback_hours,
            now=now,
            output_dir=str(args.output_dir),
            use_real_llm=args.real_llm,
        )
        print(f"wrote reflection evaluation artifact to {result.artifact_path}")
        print(f"selected channels: {len(result.input_set.selected_scopes)}")
        print(f"hourly reflections: {len(result.hourly_results)}")
        print(f"daily syntheses: {len(result.daily_results)}")
        print(f"fallback used: {result.input_set.fallback_used}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


def _parse_optional_datetime(value: str | None) -> datetime | None:
    """Parse an optional ISO datetime argument."""

    if not value:
        return_value = None
        return return_value
    normalized = value.replace("Z", "+00:00")
    return_value = datetime.fromisoformat(normalized)
    return return_value


if __name__ == "__main__":
    async_main()
