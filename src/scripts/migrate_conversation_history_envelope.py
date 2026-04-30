"""Rewrite stored conversation rows into the typed envelope storage contract.

This command is an operator-run maintenance utility. It is not imported by the
chat service startup path; runtime code expects the database to already satisfy
the strict conversation-history contract.

Typical use:
    python -m scripts.migrate_conversation_history_envelope
    python -m scripts.migrate_conversation_history_envelope --apply
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
from typing import Any

from scripts._db_export import (
    configure_logging,
    configure_stdout,
    load_project_env,
)
from kazusa_ai_chatbot.config import CHARACTER_GLOBAL_USER_ID
from kazusa_ai_chatbot.db import close_db, get_db

logger = logging.getLogger(__name__)

_CONVERSATION_MIGRATION_BATCH_SIZE = 500
_LEGACY_REPLY_MARKER_PATTERN = re.compile(r"^\s*\[Reply to message\]\s*")
_CQ_REPLY_PATTERN = re.compile(r"\[CQ:reply,id=([^\],]+)[^\]]*\]")
_CQ_AT_PATTERN = re.compile(r"\[CQ:at,qq=([^\],]+)[^\]]*\]")
_CQ_ANY_PATTERN = re.compile(r"\[CQ:[^\]]+\]")
_DISCORD_USER_MENTION_PATTERN = re.compile(r"<@!?\d+>")
_DISCORD_ROLE_MENTION_PATTERN = re.compile(r"<@&\d+>")
_DISCORD_CHANNEL_MENTION_PATTERN = re.compile(r"<#\d+>")
_DISCORD_CUSTOM_EMOJI_PATTERN = re.compile(r"<a?:[A-Za-z0-9_]+:\d+>")
_DISCORD_EVERYONE_PATTERN = re.compile(r"@(everyone|here)\b")


def legacy_conversation_query() -> dict[str, Any]:
    """Build the selector for rows outside the typed storage contract.

    Returns:
        MongoDB filter selecting conversation rows missing typed envelope
        fields or still retaining the deprecated `content` field.
    """

    query: dict[str, Any] = {
        "$or": [
            {"content": {"$exists": True}},
            {"body_text": {"$exists": False}},
            {"raw_wire_text": {"$exists": False}},
            {"addressed_to_global_user_ids": {"$exists": False}},
            {"mentions": {"$exists": False}},
            {"broadcast": {"$exists": False}},
            {"attachments": {"$exists": False}},
        ]
    }
    return query


def normalize_legacy_body_text(raw_wire_text: str) -> str:
    """Strip known transport markers from persisted message text.

    Args:
        raw_wire_text: Original stored wire-ish text from a conversation row.

    Returns:
        Content-only text suitable for the typed `body_text` field.
    """

    body_text = _LEGACY_REPLY_MARKER_PATTERN.sub(" ", raw_wire_text)
    body_text = _CQ_REPLY_PATTERN.sub(" ", body_text)
    body_text = _CQ_AT_PATTERN.sub(" ", body_text)
    body_text = _CQ_ANY_PATTERN.sub(" ", body_text)
    body_text = _DISCORD_USER_MENTION_PATTERN.sub(" ", body_text)
    body_text = _DISCORD_ROLE_MENTION_PATTERN.sub(" ", body_text)
    body_text = _DISCORD_CHANNEL_MENTION_PATTERN.sub(" ", body_text)
    body_text = _DISCORD_CUSTOM_EMOJI_PATTERN.sub(" ", body_text)
    body_text = _DISCORD_EVERYONE_PATTERN.sub(" ", body_text)
    without_runs = re.sub(r"[ \t]+", " ", body_text)
    normalized_lines = [
        line.strip()
        for line in without_runs.splitlines()
    ]
    without_empty_runs = "\n".join(normalized_lines)
    return_value = re.sub(r"\n{3,}", "\n\n", without_empty_runs).strip()
    return return_value


def legacy_raw_wire_text(row: dict[str, Any]) -> str:
    """Choose the best available original text for a stored row.

    Args:
        row: Raw MongoDB conversation document.

    Returns:
        Text to retain in `raw_wire_text`.
    """

    for field in ("raw_wire_text", "content", "body_text"):
        value = row.get(field)
        if isinstance(value, str):
            return_value = value
            return return_value

    return_value = ""
    return return_value


def legacy_addressed_to(row: dict[str, Any]) -> list[str]:
    """Derive only deterministic addressees for a stored row.

    Args:
        row: Raw MongoDB conversation document.

    Returns:
        Typed addressees that are safe to persist without guessing from
        interleaved group chat.
    """

    addressed_to = row.get("addressed_to_global_user_ids")
    if isinstance(addressed_to, list):
        return_value = [
            str(global_user_id)
            for global_user_id in addressed_to
            if str(global_user_id).strip()
        ]
        return return_value

    role = row.get("role")
    channel_type = row.get("channel_type")
    if role == "user" and channel_type == "private":
        return_value = [CHARACTER_GLOBAL_USER_ID]
        return return_value

    reply_context = row.get("reply_context")
    if isinstance(reply_context, dict):
        reply_to_current_bot = reply_context.get("reply_to_current_bot")
        if reply_to_current_bot:
            return_value = [CHARACTER_GLOBAL_USER_ID]
            return return_value

    return_value: list[str] = []
    return return_value


def legacy_list_field(row: dict[str, Any], field: str) -> list[Any]:
    """Return a list field only when it already has list shape.

    Args:
        row: Raw MongoDB conversation document.
        field: Field name to inspect.

    Returns:
        Existing list value or an empty list for malformed or missing data.
    """

    value = row.get(field)
    if isinstance(value, list):
        return_value = value
        return return_value

    return_value: list[Any] = []
    return return_value


def legacy_conversation_update(row: dict[str, Any]) -> dict[str, Any]:
    """Build the typed-envelope update for one stored conversation row.

    Args:
        row: Raw MongoDB conversation document.

    Returns:
        MongoDB update document that fills typed fields and removes `content`.
    """

    raw_wire_text = legacy_raw_wire_text(row)
    body_text = row.get("body_text")
    if not isinstance(body_text, str):
        body_text = normalize_legacy_body_text(raw_wire_text)

    broadcast = row.get("broadcast")
    if not isinstance(broadcast, bool):
        broadcast = False

    update = {
        "$set": {
            "body_text": body_text,
            "raw_wire_text": raw_wire_text,
            "addressed_to_global_user_ids": legacy_addressed_to(row),
            "mentions": legacy_list_field(row, "mentions"),
            "broadcast": broadcast,
            "attachments": legacy_list_field(row, "attachments"),
        },
        "$unset": {
            "content": "",
        },
    }
    return update


async def migrate_legacy_conversation_history_rows(
    *,
    dry_run: bool,
    batch_size: int = _CONVERSATION_MIGRATION_BATCH_SIZE,
) -> int:
    """Rewrite conversation rows to the typed storage contract.

    Args:
        dry_run: When true, only count matching rows and write nothing.
        batch_size: Maximum number of rows to rewrite per database batch.

    Returns:
        Number of rows counted in dry-run mode or updated in apply mode.

    Raises:
        RuntimeError: If rows still violate the typed contract after writes.
    """

    db = await get_db()
    query = legacy_conversation_query()

    if dry_run:
        legacy_count = await db.conversation_history.count_documents(query)
        print(f"{legacy_count} conversation_history row(s) need migration")
        return_value = legacy_count
        return return_value

    migrated_count = 0
    while True:
        cursor = (
            db.conversation_history
            .find(query)
            .sort("timestamp", 1)
            .limit(batch_size)
        )
        rows = await cursor.to_list(length=batch_size)
        if not rows:
            break

        for row in rows:
            update = legacy_conversation_update(row)
            await db.conversation_history.update_one({"_id": row["_id"]}, update)
            migrated_count += 1

    remaining_count = await db.conversation_history.count_documents(query)
    if remaining_count:
        raise RuntimeError(
            "conversation_history typed-envelope rewrite incomplete: "
            f"{remaining_count} row(s) still need migration"
        )

    logger.info(
        f"Migrated {migrated_count} conversation_history row(s) "
        "to typed envelope storage"
    )
    print(
        f"Migrated {migrated_count} conversation_history row(s) "
        "to typed envelope storage"
    )
    return_value = migrated_count
    return return_value


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for the conversation-history rewrite.
    """

    parser = argparse.ArgumentParser(
        description="Rewrite conversation_history rows to typed envelope storage."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write updates. Without this flag, only count matching rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_CONVERSATION_MIGRATION_BATCH_SIZE,
        help="Maximum number of rows to update per database batch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project database logs.",
    )
    return parser


async def main() -> None:
    """Run the conversation-history rewrite CLI.

    Returns:
        None.
    """

    configure_stdout()
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    try:
        await migrate_legacy_conversation_history_rows(
            dry_run=not args.apply,
            batch_size=args.batch_size,
        )
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
