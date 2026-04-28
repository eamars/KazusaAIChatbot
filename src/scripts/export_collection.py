"""Export arbitrary read-only MongoDB collection data to JSON.

Typical use:
    python -m scripts.export_collection conversation_history --filter "{\"platform_channel_id\":\"673225019\"}"
    python -m scripts.export_collection user_profiles --limit 10 --sort "{\"global_user_id\":1}"
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
    parse_json_object,
    projection_from_exclusions,
    write_json_export,
)
from kazusa_ai_chatbot.db import close_db, get_db


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for arbitrary collection export.
    """
    parser = argparse.ArgumentParser(description="Export arbitrary collection rows with a JSON filter.")
    parser.add_argument("collection", help="Collection name to read.")
    parser.add_argument("--filter", default="{}", help="MongoDB find filter as a JSON object.")
    parser.add_argument("--sort", default="{}", help="MongoDB sort document as a JSON object.")
    parser.add_argument("--limit", type=int, default=100, help="Maximum rows to export.")
    parser.add_argument(
        "--exclude-field",
        action="append",
        default=[],
        help="Additional field to exclude. Can be repeated.",
    )
    parser.add_argument("--include-default-large-fields", action="store_true", help="Keep _id and embedding fields.")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


async def main() -> None:
    """Run the arbitrary collection export CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    filter_doc = parse_json_object(args.filter, "--filter")
    sort_doc = parse_json_object(args.sort, "--sort")
    exclude_fields = list(args.exclude_field)
    if not args.include_default_large_fields:
        exclude_fields = list(DEFAULT_EXCLUDED_FIELDS) + exclude_fields
    output_path = args.output or default_output_path("collection", args.collection)

    try:
        db = await get_db()
        cursor = db[args.collection].find(
            filter_doc,
            projection_from_exclusions(exclude_fields),
        )
        if sort_doc:
            cursor = cursor.sort(list(sort_doc.items()))
        cursor = cursor.limit(args.limit)
        records = [dict(doc) for doc in await cursor.to_list(length=args.limit)]
        query = {
            "collection": args.collection,
            "filter": filter_doc,
            "sort": sort_doc,
            "limit": args.limit,
        }
        write_json_export(
            output_path=output_path,
            query=query,
            records_key="documents",
            records=records,
            exclude_fields=exclude_fields,
        )
        print(f"wrote {len(records)} document(s) to {output_path}")
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
