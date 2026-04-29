"""Export only a user's stored image bundle from MongoDB to JSON.

Typical use:
    python -m scripts.export_user_image 3167827653 --platform qq
    python -m scripts.export_user_image 263c883d-aeff-4e0b-a758-6f69186ae8ec
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Any

from scripts._db_export import (
    configure_logging,
    configure_stdout,
    default_output_path,
    load_project_env,
    write_json_export,
)
from kazusa_ai_chatbot.db import close_db, get_db


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for user-image export.
    """
    parser = argparse.ArgumentParser(description="Export user_image for one profile.")
    parser.add_argument("identifier", help="Global user id, or platform user id when --platform is set.")
    parser.add_argument("--platform", help="Platform name for platform-account lookup.")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


async def _find_profile_header(identifier: str, platform: str | None) -> dict[str, Any]:
    """Find profile identity and image fields.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is set.
        platform: Optional platform name for exact platform-account lookup.

    Returns:
        Profile subset containing identity fields and ``user_image``.
    """
    db = await get_db()
    projection = {
        "_id": 0,
        "global_user_id": 1,
        "platform_accounts": 1,
        "affinity": 1,
        "last_relationship_insight": 1,
        "user_image": 1,
    }
    if not platform:
        profile = await db.user_profiles.find_one({"global_user_id": identifier}, projection)
        if profile is not None:
            return_value = dict(profile)
            return return_value

    account_filter: dict[str, Any] = {"platform_user_id": identifier}
    if platform:
        account_filter["platform"] = platform
    profile = await db.user_profiles.find_one(
        {"platform_accounts": {"$elemMatch": account_filter}},
        projection,
    )
    return_value = dict(profile or {})
    return return_value


async def main() -> None:
    """Run the user-image export CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    output_path = args.output or default_output_path("user_image", args.identifier)

    try:
        profile = await _find_profile_header(args.identifier, args.platform)
        query = {
            "collection": "user_profiles",
            "identifier": args.identifier,
            "platform": args.platform,
            "field": "user_image",
        }
        write_json_export(
            output_path=output_path,
            query=query,
            records_key="profile",
            records=profile,
            exclude_fields=["_id"],
        )
        print(f"wrote user image to {output_path}")
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
