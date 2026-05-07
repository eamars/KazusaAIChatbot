"""Export shared ``memory`` collection rows from MongoDB to JSON.

Typical use:
    python -m scripts.export_memory --source-global-user-id 263c883d-aeff-4e0b-a758-6f69186ae8ec
    python -m scripts.export_memory --memory-type promise --status active
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from scripts._db_export import (
    DEFAULT_EXCLUDED_FIELDS,
    configure_logging,
    configure_stdout,
    default_output_path,
    load_project_env,
    projection_from_exclusions,
    write_json_export,
)
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import export_memory_rows


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for shared-memory export.
    """
    parser = argparse.ArgumentParser(description="Export rows from the shared memory collection.")
    parser.add_argument("--memory-unit-id", help="Filter by memory_unit_id.")
    parser.add_argument("--lineage-id", help="Filter by lineage_id.")
    parser.add_argument("--source-global-user-id", help="Filter by source_global_user_id.")
    parser.add_argument("--memory-type", help="Filter by memory_type.")
    parser.add_argument("--source-kind", help="Filter by source_kind.")
    parser.add_argument("--authority", help="Filter by authority.")
    parser.add_argument("--status", help="Filter by status.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to export.")
    parser.add_argument("--include-embeddings", action="store_true", help="Include vector embeddings in output.")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


def _build_query(args: argparse.Namespace) -> dict[str, Any]:
    """Build the MongoDB memory filter from parsed arguments.

    Args:
        args: Parsed command-line namespace.

    Returns:
        MongoDB query for the memory collection.
    """
    query: dict[str, Any] = {}
    for argument_name in (
        "memory_unit_id",
        "lineage_id",
        "source_global_user_id",
        "memory_type",
        "source_kind",
        "authority",
        "status",
    ):
        value = getattr(args, argument_name)
        if value:
            query[argument_name] = value
    return query


async def main() -> None:
    """Run the shared-memory export CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    exclude_fields = [] if args.include_embeddings else list(DEFAULT_EXCLUDED_FIELDS)
    query_filter = _build_query(args)
    identifier = (
        args.memory_unit_id
        or args.lineage_id
        or args.source_global_user_id
        or args.memory_type
        or "all"
    )
    output_path = args.output or default_output_path("memory", identifier)

    try:
        records = await export_memory_rows(
            query_filter=query_filter,
            projection=projection_from_exclusions(exclude_fields),
            limit=args.limit,
        )
        query = {
            "collection": "memory",
            "filter": query_filter,
            "limit": args.limit,
            "sort": "updated_at descending, timestamp descending",
        }
        write_json_export(
            output_path=output_path,
            query=query,
            records_key="memories",
            records=records,
            exclude_fields=exclude_fields,
        )
        print(f"wrote {len(records)} memory row(s) to {output_path}")
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
