"""Export the singleton character_state document from MongoDB to JSON.

Typical use:
    python -m scripts.export_character_state
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts._db_export import (
    DEFAULT_EXCLUDED_FIELDS,
    configure_logging,
    configure_stdout,
    default_output_path,
    load_project_env,
    write_json_export,
)
from kazusa_ai_chatbot.db import close_db, get_character_profile


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for character-state export.
    """
    parser = argparse.ArgumentParser(description="Export character_state/_id:global.")
    parser.add_argument("--include-embeddings", action="store_true", help="Include vector embeddings in output.")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


async def main() -> None:
    """Run the character-state export CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    exclude_fields = [] if args.include_embeddings else list(DEFAULT_EXCLUDED_FIELDS)
    output_path = args.output or default_output_path("character_state", "global")

    try:
        state = dict(await get_character_profile())
        query = {
            "collection": "character_state",
            "_id": "global",
        }
        write_json_export(
            output_path=output_path,
            query=query,
            records_key="character_state",
            records=state,
            exclude_fields=exclude_fields,
        )
        print(f"wrote character state to {output_path}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI.

    Returns:
        None.
    """
    asyncio.run(main())


if __name__ == "__main__":
    async_main()
