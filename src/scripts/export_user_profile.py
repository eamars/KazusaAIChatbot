"""Export a user profile from MongoDB to JSON.

Typical use:
    python -m scripts.export_user_profile 3167827653 --platform qq
    python -m scripts.export_user_profile 263c883d-aeff-4e0b-a758-6f69186ae8ec
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
    write_json_export,
)
from kazusa_ai_chatbot.db import close_db, get_db, get_user_profile


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for user-profile export.
    """
    parser = argparse.ArgumentParser(description="Export one user_profiles document.")
    parser.add_argument("identifier", help="Global user id, or platform user id when --platform is set.")
    parser.add_argument("--platform", help="Platform name for platform-account lookup.")
    parser.add_argument("--include-embeddings", action="store_true", help="Include vector embeddings in output.")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


async def _find_profile(identifier: str, platform: str | None) -> dict[str, Any]:
    """Find a profile by global id or platform account id.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is set.
        platform: Optional platform name for exact platform-account lookup.

    Returns:
        Matching profile document without ``_id`` when found, otherwise an empty dict.
    """
    if not platform:
        profile = await get_user_profile(identifier)
        if profile:
            return dict(profile)

    db = await get_db()
    account_filter: dict[str, Any] = {"platform_user_id": identifier}
    if platform:
        account_filter["platform"] = platform
    doc = await db.user_profiles.find_one(
        {"platform_accounts": {"$elemMatch": account_filter}},
        {"_id": 0},
    )
    return dict(doc or {})


async def main() -> None:
    """Run the user-profile export CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    exclude_fields = [] if args.include_embeddings else list(DEFAULT_EXCLUDED_FIELDS)
    output_path = args.output or default_output_path("user_profile", args.identifier)

    try:
        profile = await _find_profile(args.identifier, args.platform)
        query = {
            "collection": "user_profiles",
            "identifier": args.identifier,
            "platform": args.platform,
        }
        write_json_export(
            output_path=output_path,
            query=query,
            records_key="profile",
            records=profile,
            exclude_fields=exclude_fields,
        )
        print(f"wrote user profile to {output_path}")
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
