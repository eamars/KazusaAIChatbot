"""CLI helper to inspect a user's cognition-facing profile bundle.

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
    empty_interaction_style_overlay,
    find_user_profile_by_identifier,
    get_user_style_image,
)
from kazusa_ai_chatbot.rag.user_memory_unit_retrieval import build_user_memory_context


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
    return_value = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return return_value


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
    return_value = "\n".join(lines)
    return return_value


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
    return_value = "\n".join(lines)
    return return_value


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

    hydrated = dict(profile)
    hydrated["user_memory_context"] = await build_user_memory_context(
        global_user_id,
        query_text="",
        include_semantic=False,
    )
    style_doc = await get_user_style_image(global_user_id)
    if style_doc is not None:
        hydrated["user_style_image"] = {
            "status": style_doc["status"],
            "overlay": style_doc["overlay"],
            "revision": style_doc["revision"],
            "updated_at": style_doc["updated_at"],
        }
    else:
        hydrated["user_style_image"] = {
            "status": "none",
            "overlay": empty_interaction_style_overlay(),
            "revision": "",
            "updated_at": "",
        }
    return hydrated


def _format_profile(profile: dict[str, Any]) -> str:
    """Format a matched profile's image bundle for terminal output.

    Args:
        profile: User profile document returned from MongoDB without ``_id``.

    Returns:
        Human-readable profile and user-image summary.
    """
    accounts = profile.get("platform_accounts") or []
    if not isinstance(accounts, list):
        accounts = []

    relationship = str(profile.get("last_relationship_insight", "")).strip()
    if not relationship:
        relationship = "none"

    user_memory_context = profile.get("user_memory_context") or {}
    if not isinstance(user_memory_context, dict):
        user_memory_context = {}
    user_style_image = profile.get("user_style_image") or {}
    if not isinstance(user_style_image, dict):
        user_style_image = {}
    style_overlay = user_style_image.get("overlay") or {}
    if not isinstance(style_overlay, dict):
        style_overlay = {}

    return_value = "\n".join(
        [
            f"global_user_id: {profile.get('global_user_id', '')}",
            "platform_accounts:",
            _format_accounts(accounts),
            f"affinity: {profile.get('affinity', '')}",
            f"last_relationship_insight: {relationship}",
            "",
            "user_memory_context.stable_patterns:",
            _format_sequence(user_memory_context.get("stable_patterns") or [], label="stable"),
            "",
            "user_memory_context.recent_shifts:",
            _format_sequence(user_memory_context.get("recent_shifts") or [], label="recent-shift"),
            "",
            "user_memory_context.objective_facts:",
            _format_sequence(user_memory_context.get("objective_facts") or [], label="fact"),
            "",
            "user_memory_context.milestones:",
            _format_sequence(user_memory_context.get("milestones") or [], label="milestone"),
            "",
            "user_memory_context.active_commitments:",
            _format_sequence(
                user_memory_context.get("active_commitments") or [],
                label="commitment",
            ),
            "",
            "user_style_image:",
            f"  status: {user_style_image.get('status', 'none')}",
            f"  revision: {user_style_image.get('revision', '')}",
            f"  updated_at: {user_style_image.get('updated_at', '')}",
            "user_style_image.speech_guidelines:",
            _format_sequence(
                style_overlay.get("speech_guidelines") or [],
                label="speech",
            ),
            "",
            "user_style_image.social_guidelines:",
            _format_sequence(
                style_overlay.get("social_guidelines") or [],
                label="social",
            ),
            "",
            "user_style_image.pacing_guidelines:",
            _format_sequence(
                style_overlay.get("pacing_guidelines") or [],
                label="pacing",
            ),
            "",
            "user_style_image.engagement_guidelines:",
            _format_sequence(
                style_overlay.get("engagement_guidelines") or [],
                label="engagement",
            ),
        ]
    )
    return return_value


async def _find_profile(identifier: str, platform: str | None) -> dict[str, Any] | None:
    """Find a profile by global id or platform account id.

    Args:
        identifier: Global user id, or platform user id when ``platform`` is set.
        platform: Optional platform name for exact platform-account lookup.

    Returns:
        The first matching profile document with ``_id`` omitted, or ``None``.
    """
    profile = await find_user_profile_by_identifier(
        identifier=identifier,
        platform=platform,
    )
    if profile is None:
        return_value = None
        return return_value
    return_value = dict(profile)
    return return_value


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
