"""Repair stored conversation rows for the typed envelope storage contract.

This command is an operator-run maintenance utility. It is not imported by the
chat service startup path; runtime code expects the database to already satisfy
the strict conversation-history contract. It fills missing typed fields on
legacy rows and removes transport syntax that leaked into semantic text fields.

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
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import (
    count_legacy_conversation_history_rows,
    list_legacy_conversation_history_rows,
    update_conversation_history_row,
)

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
BODY_TEXT_TRANSPORT_SYNTAX_PATTERN = (
    r"(\[Reply to message\]|\[CQ:[^\]]+\]|<@!?\d+>|<@&\d+>|"
    r"<#\d+>|<a?:[A-Za-z0-9_]+:\d+>)"
)
_BODY_TEXT_TRANSPORT_PATTERNS = (
    _LEGACY_REPLY_MARKER_PATTERN,
    _CQ_ANY_PATTERN,
    _DISCORD_USER_MENTION_PATTERN,
    _DISCORD_ROLE_MENTION_PATTERN,
    _DISCORD_CHANNEL_MENTION_PATTERN,
    _DISCORD_CUSTOM_EMOJI_PATTERN,
)


def _collapse_legacy_text_spacing(text: str) -> str:
    """Collapse whitespace after transport markers are removed."""

    without_runs = re.sub(r"[ \t]+", " ", text)
    normalized_lines = [
        line.strip()
        for line in without_runs.splitlines()
    ]
    without_empty_runs = "\n".join(normalized_lines)
    return_value = re.sub(r"\n{3,}", "\n\n", without_empty_runs).strip()
    return return_value


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
    return_value = _collapse_legacy_text_spacing(body_text)
    return return_value


def body_text_has_transport_syntax(body_text: str) -> bool:
    """Return whether a typed text field still contains transport syntax."""

    for pattern in _BODY_TEXT_TRANSPORT_PATTERNS:
        if pattern.search(body_text):
            return_value = True
            return return_value

    return_value = False
    return return_value


def normalize_dirty_body_text(body_text: str) -> str:
    """Strip transport syntax from an existing typed text field.

    Args:
        body_text: Stored semantic text that may still contain platform
            transport markers.

    Returns:
        Cleaned text for the typed storage contract. Readable broadcast tokens
        such as ``@everyone`` are preserved because they are valid body text.
    """

    clean_text = _LEGACY_REPLY_MARKER_PATTERN.sub(" ", body_text)
    clean_text = _CQ_REPLY_PATTERN.sub(" ", clean_text)
    clean_text = _CQ_AT_PATTERN.sub(" ", clean_text)
    clean_text = _CQ_ANY_PATTERN.sub(" ", clean_text)
    clean_text = _DISCORD_USER_MENTION_PATTERN.sub(" ", clean_text)
    clean_text = _DISCORD_ROLE_MENTION_PATTERN.sub(" ", clean_text)
    clean_text = _DISCORD_CHANNEL_MENTION_PATTERN.sub(" ", clean_text)
    clean_text = _DISCORD_CUSTOM_EMOJI_PATTERN.sub(" ", clean_text)
    return_value = _collapse_legacy_text_spacing(clean_text)
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


def sanitized_reply_context(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return a repaired reply context only when one needs text cleanup.

    Args:
        row: Raw MongoDB conversation document.

    Returns:
        A reply-context copy with dirty excerpt text repaired, or ``None`` when
        the row does not need a reply-context update.
    """

    reply_context = row.get("reply_context")
    if not isinstance(reply_context, dict):
        return_value: dict[str, Any] | None = None
        return return_value

    reply_excerpt = reply_context.get("reply_excerpt")
    if not isinstance(reply_excerpt, str):
        return_value = None
        return return_value

    if not body_text_has_transport_syntax(reply_excerpt):
        return_value = None
        return return_value

    repaired_context = dict(reply_context)
    repaired_excerpt = normalize_dirty_body_text(reply_excerpt)
    if repaired_excerpt:
        repaired_context["reply_excerpt"] = repaired_excerpt
    else:
        repaired_context.pop("reply_excerpt", None)

    return_value = repaired_context
    return return_value


def legacy_conversation_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Build typed-envelope fields for one stored conversation row.

    Args:
        row: Raw MongoDB conversation document.

    Returns:
        Field values that fill the typed conversation storage contract.
    """

    raw_wire_text = legacy_raw_wire_text(row)
    body_text = row.get("body_text")
    if isinstance(body_text, str):
        if body_text_has_transport_syntax(body_text):
            body_text = normalize_dirty_body_text(body_text)
    else:
        body_text = normalize_legacy_body_text(raw_wire_text)

    broadcast = row.get("broadcast")
    if not isinstance(broadcast, bool):
        broadcast = False

    fields = {
        "body_text": body_text,
        "raw_wire_text": raw_wire_text,
        "addressed_to_global_user_ids": legacy_addressed_to(row),
        "mentions": legacy_list_field(row, "mentions"),
        "broadcast": broadcast,
        "attachments": legacy_list_field(row, "attachments"),
    }
    reply_context = sanitized_reply_context(row)
    if reply_context is not None:
        fields["reply_context"] = reply_context
    return fields


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

    if dry_run:
        legacy_count = await count_legacy_conversation_history_rows(
            semantic_text_pattern=BODY_TEXT_TRANSPORT_SYNTAX_PATTERN,
        )
        print(f"{legacy_count} conversation_history row(s) need migration")
        return_value = legacy_count
        return return_value

    migrated_count = 0
    while True:
        rows = await list_legacy_conversation_history_rows(
            batch_size=batch_size,
            semantic_text_pattern=BODY_TEXT_TRANSPORT_SYNTAX_PATTERN,
        )
        if not rows:
            break

        for row in rows:
            fields = legacy_conversation_fields(row)
            await update_conversation_history_row(
                row_id=row["_id"],
                set_fields=fields,
                unset_fields=("content",),
            )
            migrated_count += 1

    remaining_count = await count_legacy_conversation_history_rows(
        semantic_text_pattern=BODY_TEXT_TRANSPORT_SYNTAX_PATTERN,
    )
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
        description="Repair conversation_history rows for typed envelope storage."
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
