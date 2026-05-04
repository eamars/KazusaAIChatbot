"""Reset shared memory lore from repository seed data."""

from __future__ import annotations

import argparse
import asyncio
import json

from scripts._db_export import configure_logging, configure_stdout, load_project_env
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.memory_evolution import reset_memory_from_seed


def _build_parser() -> argparse.ArgumentParser:
    """Build the reset CLI parser.

    Returns:
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Reset shared memory lore from personalities/knowledge seed data.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Report changes only.")
    mode.add_argument("--apply", action="store_true", help="Apply the reset.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


async def main() -> None:
    """Run the reset CLI."""
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    try:
        result = await reset_memory_from_seed(dry_run=not args.apply)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""
    asyncio.run(main())


if __name__ == "__main__":
    async_main()
