"""CLI helper to inspect a group channel interaction-style image.

Typical use:
    python -m scripts.identify_group_image 1082431481 --platform qq
    identify-group-image 1082431481 --platform qq --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any

from kazusa_ai_chatbot.db import close_db, get_group_channel_style_image


_GUIDELINE_LABELS = (
    ("speech_guidelines", "speech"),
    ("social_guidelines", "social"),
    ("pacing_guidelines", "pacing"),
    ("engagement_guidelines", "engagement"),
)


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when the active stream supports it.

    Returns:
        None.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for group image lookup.

    Returns:
        Configured argument parser for the group style-image lookup CLI.
    """
    parser = argparse.ArgumentParser(
        description="Look up a stored group channel style image."
    )
    parser.add_argument(
        "platform_channel_id",
        help="Platform channel or group id, for example a QQ group id.",
    )
    parser.add_argument(
        "--platform",
        required=True,
        help="Platform name for the channel lookup, for example qq.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the matched group image as JSON instead of compact text.",
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


def _format_sequence(items: list[Any], *, label: str) -> str:
    """Format a list of interaction style guideline entries.

    Args:
        items: Guideline strings from the stored style image.
        label: Label to prefix each formatted guideline with.

    Returns:
        Newline-separated compact sequence summary.
    """
    if not items:
        return "  - none"

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        text = str(item).strip()
        if text:
            lines.append(f"  {index}. {label}: {text}")
    if not lines:
        return_value = "  - none"
        return return_value
    return_value = "\n".join(lines)
    return return_value


def _format_group_style_image(document: dict[str, Any]) -> str:
    """Format a group channel style image for terminal output.

    Args:
        document: Stored ``interaction_style_images`` document without Mongo
            internals.

    Returns:
        Human-readable group channel image summary.
    """
    overlay = document.get("overlay") or {}
    if not isinstance(overlay, dict):
        overlay = {}

    lines = [
        f"style_image_id: {document.get('style_image_id', '')}",
        f"platform: {document.get('platform', '')}",
        f"platform_channel_id: {document.get('platform_channel_id', '')}",
        "",
        "group_channel_style_image:",
        f"  status: {document.get('status', 'none')}",
        f"  revision: {document.get('revision', '')}",
        f"  updated_at: {document.get('updated_at', '')}",
        f"  confidence: {overlay.get('confidence', '')}",
    ]
    for field_name, label in _GUIDELINE_LABELS:
        guideline_items = overlay.get(field_name) or []
        if not isinstance(guideline_items, list):
            guideline_items = []
        lines.extend(
            [
                "",
                f"group_channel_style_image.{field_name}:",
                _format_sequence(guideline_items, label=label),
            ]
        )
    return_value = "\n".join(lines)
    return return_value


async def _find_group_style_image(
    *,
    platform: str,
    platform_channel_id: str,
) -> dict[str, Any] | None:
    """Find a group channel style image by platform scope.

    Args:
        platform: Platform namespace, such as ``qq``.
        platform_channel_id: Platform channel or group id.

    Returns:
        The matching style image document, or ``None`` when no image exists.
    """
    document = await get_group_channel_style_image(
        platform=platform,
        platform_channel_id=platform_channel_id,
    )
    if document is None:
        return_value = None
        return return_value
    return_value = dict(document)
    return return_value


async def main() -> None:
    """Run the group image lookup CLI.

    Returns:
        None.
    """
    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    platform = args.platform.strip()
    platform_channel_id = args.platform_channel_id.strip()

    try:
        document = await _find_group_style_image(
            platform=platform,
            platform_channel_id=platform_channel_id,
        )
        if document is None:
            print(
                "No group channel style image found for "
                f"{platform}:{platform_channel_id}."
            )
            return

        if args.json:
            json_output = json.dumps(
                document,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            print(json_output)
            return

        print(_format_group_style_image(document))
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
