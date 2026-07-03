"""Read-only audit for semantic identity pollution in durable storage."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from bson import json_util

from kazusa_ai_chatbot.db import close_db
from scripts._db_export import (
    configure_logging,
    configure_stdout,
    load_project_env,
)
from scripts.repair_semantic_identity_pollution import (
    DEFAULT_BATCH_SIZE,
    build_semantic_identity_report,
)


DEFAULT_OUTPUT = (
    Path("test_artifacts")
    / "diagnostics"
    / "semantic_identity_pollution_dry_run.json"
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_util.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for the read-only audit."""

    parser = argparse.ArgumentParser(
        description="Audit semantic identity pollution without mutations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Maximum rows to inspect per dirty collection.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


async def main() -> None:
    """Run the read-only semantic identity pollution audit."""

    configure_stdout()
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    try:
        report = await build_semantic_identity_report(
            dry_run=True,
            batch_size=args.batch_size,
        )
        _write_json(args.output, report)
        print(json_util.dumps(report, ensure_ascii=False, indent=2))
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
