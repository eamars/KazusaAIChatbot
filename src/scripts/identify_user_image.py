"""CLI helper to inspect a user's stored image bundle.

Typical use:
    python -m scripts.identify_user_image 3167827653 --platform qq
    python -m scripts.identify_user_image 263c883d-aeff-4e0b-a758-6f69186ae8ec
    identify-user-image 3167827653 --platform qq --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from kazusa_ai_chatbot.db import (
    close_db,
    get_db,
    hydrate_user_profile_with_memory_blocks,
    query_user_profile_memory_blocks,
)


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when the active stream supports it.

    Returns:
        None.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for the user-image lookup CLI.
    """
    parser = argparse.ArgumentParser(
        description="Look up a stored user image by global_user_id or platform account id."
    )
    parser.add_argument(
        "identifier",
        help="Global user id, or a platform user id when --platform is provided.",
    )
    parser.add_argument(
        "--platform",
        help="Platform name for a platform user id lookup, for example qq.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the matched profile as JSON instead of the compact text view.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project database logs while running the lookup.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Keep the quick lookup quiet unless verbose output is requested.

    Args:
        verbose: Whether to preserve project INFO logs.

    Returns:
        None.
    """
    if not verbose:
        logging.getLogger("kazusa_ai_chatbot").setLevel(logging.WARNING)


def _compact_json(value: Any) -> str:
    """Serialize a value as stable, readable JSON.

    Args:
        value: JSON-compatible value loaded from MongoDB.

    Returns:
        Pretty-printed JSON with non-ASCII text preserved.
    """
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _format_accounts(accounts: list[dict[str, Any]]) -> str:
    """Format linked platform accounts for terminal output.

    Args:
        accounts: Raw ``platform_accounts`` list from a user profile document.

    Returns:
        Newline-separated account summary.
    """
    if not accounts:
        return "  - none"

    lines: list[str] = []
    for account in accounts:
        platform = str(account.get("platform", "")).strip() or "unknown-platform"
        platform_user_id = str(account.get("platform_user_id", "")).strip() or "unknown-id"
        display_name = str(account.get("display_name", "")).strip()
        suffix = f" ({display_name})" if display_name else ""
        lines.append(f"  - {platform}:{platform_user_id}{suffix}")
    return "\n".join(lines)


def _format_sequence(items: list[Any], *, label: str) -> str:
    """Format a list of prompt-facing profile entries.

    Args:
        items: Profile sequence from MongoDB or hydrated memory blocks.
        label: Fallback label to show when an item lacks a known text field.

    Returns:
        Newline-separated compact sequence summary.
    """
    if not items:
        return "  - none"

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        if isinstance(item, dict):
            text = (
                str(item.get("summary", "")).strip()
                or str(item.get("event", "")).strip()
                or str(item.get("description", "")).strip()
                or str(item.get("entry", "")).strip()
                or str(item.get("fact", "")).strip()
                or str(item.get("action", "")).strip()
                or _compact_json(item)
            )
            timestamp = str(item.get("timestamp", "")).strip() or str(item.get("created_at", "")).strip()
            prefix = f"{timestamp} | " if timestamp else ""
            lines.append(f"  {index}. {prefix}{text}")
        else:
            lines.append(f"  {index}. {label}: {item}")
    return "\n".join(lines)


async def _hydrate_prompt_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Hydrate a stored profile into the prompt-facing cognition bundle.

    Args:
        profile: Raw ``user_profiles`` document without ``_id``.

    Returns:
        Profile merged with prompt-facing memory blocks, matching the
        deterministic nonsemantic shape used by the RAG profile read path.
    """
    global_user_id = str(profile.get("global_user_id", "")).strip()
    if not global_user_id:
        return profile

    memory_blocks = await query_user_profile_memory_blocks(global_user_id, include_semantic=False)
    return hydrate_user_profile_with_memory_blocks(profile, memory_blocks)


def _format_profile(profile: dict[str, Any]) -> str:
    """Format a matched profile's image bundle for terminal output.

    Args:
        profile: User profile document returned from MongoDB without ``_id``.

    Returns:
        Human-readable profile and user-image summary.
    """
    image = profile.get("user_image") or {}
    if not isinstance(image, dict):
        image = {"raw_user_image": image}

    accounts = profile.get("platform_accounts") or []
    if not isinstance(accounts, list):
        accounts = []

    milestones = image.get("milestones") or []
    if not isinstance(milestones, list):
        milestones = [milestones]

    recent_window = image.get("recent_window") or []
    if not isinstance(recent_window, list):
        recent_window = [recent_window]

    historical_summary = str(image.get("historical_summary", "")).strip()
    if not historical_summary:
        historical_summary = "none"

    relationship = str(profile.get("last_relationship_insight", "")).strip()
    if not relationship:
        relationship = "none"

    character_diary = profile.get("character_diary") or []
    if not isinstance(character_diary, list):
        character_diary = [character_diary]

    objective_facts = profile.get("objective_facts") or []
    if not isinstance(objective_facts, list):
        objective_facts = [objective_facts]

    active_commitments = profile.get("active_commitments") or []
    if not isinstance(active_commitments, list):
        active_commitments = [active_commitments]

    return "\n".join(
        [
            f"global_user_id: {profile.get('global_user_id', '')}",
            "platform_accounts:",
            _format_accounts(accounts),
            f"affinity: {profile.get('affinity', '')}",
            f"last_relationship_insight: {relationship}",
            "",
            "user_image.historical_summary:",
            f"  {historical_summary}",
            "",
            "user_image.recent_window:",
            _format_sequence(recent_window, label="recent"),
            "",
            "user_image.milestones:",
            _format_sequence(milestones, label="milestone"),
            "",
            "character_diary:",
            _format_sequence(character_diary, label="diary"),
            "",
            "objective_facts:",
            _format_sequence(objective_facts, label="fact"),
            "",
            "active_commitments:",
            _format_sequence(active_commitments, label="commitment"),
        ]
    )


async def _find_profile(identifier: str, platform: str | None) -> dict[str, Any] | None:
    """Find a profile by global id or platform account id.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is set.
        platform: Optional platform name for exact platform-account lookup.

    Returns:
        The first matching profile document with ``_id`` omitted, or ``None``.
    """
    db = await get_db()
    projection = {"_id": 0}
    if platform:
        return await db.user_profiles.find_one(
            {
                "platform_accounts": {
                    "$elemMatch": {
                        "platform": platform,
                        "platform_user_id": identifier,
                    }
                }
            },
            projection,
        )

    profile = await db.user_profiles.find_one({"global_user_id": identifier}, projection)
    if profile is not None:
        return profile

    return await db.user_profiles.find_one(
        {"platform_accounts": {"$elemMatch": {"platform_user_id": identifier}}},
        projection,
    )


async def main() -> None:
    """Run the user-image lookup CLI.

    Returns:
        None.
    """
    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    identifier = args.identifier.strip()
    platform = args.platform.strip() if args.platform else None

    try:
        profile = await _find_profile(identifier, platform)
        if profile is None:
            lookup_hint = f"{platform}:{identifier}" if platform else identifier
            print(f"No user profile found for {lookup_hint}.")
            return

        profile = await _hydrate_prompt_profile(dict(profile))

        if args.json:
            print(_compact_json(profile))
            return

        print(_format_profile(profile))
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
