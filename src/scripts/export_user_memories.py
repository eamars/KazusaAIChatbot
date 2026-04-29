"""Export persistent user memory units for one user from MongoDB to JSON.

Typical use:
    python -m scripts.export_user_memories 263c883d-aeff-4e0b-a758-6f69186ae8ec
    python -m scripts.export_user_memories 3167827653 --platform qq --raw
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
from kazusa_ai_chatbot.db import (
    close_db,
    get_db,
    get_user_profile,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import build_user_memory_context


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for user-memory export.
    """
    parser = argparse.ArgumentParser(description="Export persistent user_memory_units for one user.")
    parser.add_argument("identifier", help="Global user id, or platform user id when --platform is set.")
    parser.add_argument("--platform", help="Platform name for platform-account lookup.")
    parser.add_argument("--raw", action="store_true", help="Export raw memory documents instead of prompt-facing blocks.")
    parser.add_argument("--include-inactive", action="store_true", help="Include archived/completed/cancelled rows in raw mode.")
    parser.add_argument("--include-embeddings", action="store_true", help="Include vector embeddings in raw output.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum raw memory rows to export.")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


async def _resolve_identifier(identifier: str, platform: str | None) -> str:
    """Resolve a global user id from a global id or platform account.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is set.
        platform: Optional platform name for exact platform-account lookup.

    Returns:
        Resolved global user id, or an empty string when not found.
    """
    if not platform:
        profile = await get_user_profile(identifier)
        if profile:
            return str(profile["global_user_id"])

    db = await get_db()
    account_filter: dict[str, Any] = {"platform_user_id": identifier}
    if platform:
        account_filter["platform"] = platform
    profile = await db.user_profiles.find_one(
        {"platform_accounts": {"$elemMatch": account_filter}},
        {"_id": 0, "global_user_id": 1},
    )
    return str((profile or {}).get("global_user_id", ""))


async def _load_raw_units(
    global_user_id: str,
    *,
    include_inactive: bool,
    exclude_fields: list[str],
    limit: int,
) -> list[dict[str, Any]]:
    """Load raw ``user_memory_units`` rows.

    Args:
        global_user_id: Owner of the memory units.
        include_inactive: Whether inactive rows should be included.
        exclude_fields: Field names to exclude from MongoDB projection.
        limit: Maximum number of rows.

    Returns:
        Raw memory-unit documents sorted by newest first.
    """
    db = await get_db()
    query: dict[str, Any] = {"global_user_id": global_user_id}
    if not include_inactive:
        query["status"] = "active"
    cursor = (
        db.user_memory_units
        .find(query, projection_from_exclusions(exclude_fields))
        .sort([("last_seen_at", -1), ("updated_at", -1)])
        .limit(limit)
    )
    return [dict(doc) for doc in await cursor.to_list(length=limit)]


async def main() -> None:
    """Run the user-memory export CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    exclude_fields = [] if args.include_embeddings else list(DEFAULT_EXCLUDED_FIELDS)
    output_path = args.output or default_output_path("user_memories", args.identifier)

    try:
        global_user_id = await _resolve_identifier(args.identifier, args.platform)
        if args.raw:
            records_key = "memories"
            records: list[dict[str, Any]] | dict[str, Any] = []
            if global_user_id:
                records = await _load_raw_units(
                    global_user_id,
                    include_inactive=args.include_inactive,
                    exclude_fields=exclude_fields,
                    limit=args.limit,
                )
        else:
            records_key = "user_memory_context"
            records = {}
            if global_user_id:
                records = await build_user_memory_context(
                    global_user_id,
                    query_text="",
                    include_semantic=False,
                )

        query = {
            "collection": "user_memory_units",
            "identifier": args.identifier,
            "platform": args.platform,
            "global_user_id": global_user_id,
            "raw": args.raw,
            "include_inactive": args.include_inactive,
            "limit": args.limit,
        }
        write_json_export(
            output_path=output_path,
            query=query,
            records_key=records_key,
            records=records,
            exclude_fields=exclude_fields,
        )
        if isinstance(records, list):
            print(f"wrote {len(records)} memory unit(s) to {output_path}")
        else:
            print(f"wrote user memory context to {output_path}")
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
