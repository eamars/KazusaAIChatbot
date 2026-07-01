"""Apply or dry-run TTL expiry for existing logging rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import (
    apply_logging_retention,
)


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when available."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Apply TTL expiry fields to logging collections.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser


async def main() -> None:
    """Run the retention maintenance CLI."""

    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        report = await apply_logging_retention(
            dry_run=bool(args.dry_run),
            batch_size=args.batch_size,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
