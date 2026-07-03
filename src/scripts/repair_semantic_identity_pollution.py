"""Audit and repair semantic identity pollution in durable storage."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from bson import json_util

from adapters.envelope_common import (
    normalize_mention_display_label,
    semantic_entity_fallback_label,
)
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import (
    SEMANTIC_IDENTITY_FORBIDDEN_PATTERN,
    archive_user_memory_unit_for_semantic_identity_repair,
    count_semantic_identity_conversation_rows,
    count_semantic_identity_shared_memory_units,
    count_semantic_identity_user_memory_units,
    count_semantic_identity_user_profile_accounts,
    list_semantic_identity_conversation_rows,
    list_semantic_identity_shared_memory_units,
    list_semantic_identity_user_memory_units,
    list_semantic_identity_user_profile_rows,
    update_conversation_history_row_for_semantic_identity_repair,
    update_user_profile_platform_account_display_name,
)
from kazusa_ai_chatbot.memory_evolution import reject_memory_unit
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from scripts._db_export import (
    configure_logging,
    configure_stdout,
    load_project_env,
)
from scripts.migrate_conversation_history_envelope import normalize_dirty_body_text


DEFAULT_BATCH_SIZE = 1000
DEFAULT_DRY_RUN_OUTPUT = (
    Path("test_artifacts")
    / "diagnostics"
    / "semantic_identity_repair_dry_run.json"
)
_FORBIDDEN_PATTERN = re.compile(SEMANTIC_IDENTITY_FORBIDDEN_PATTERN)
_LEGACY_PLACEHOLDER_PATTERN = re.compile(
    r"(@mentioned-(?:user|role|entity)-\d+|#mentioned-channel-\d+)"
)
_PLATFORM_QUALIFIED_FALLBACK_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?P<prefix>[@#]?)"
    r"(?:qq|discord|platform)-"
    r"(?P<kind>user|bot|role|channel|entity):[^\s]+"
)


@dataclass
class ConversationRepair:
    """Deterministic repair result for one conversation row."""

    repairable: bool
    set_fields: dict[str, Any]
    unset_fields: tuple[str, ...]
    requires_reembedding: bool
    ambiguous_fields: list[str]


def semantic_field_is_dirty(value: object) -> bool:
    """Return whether a stored semantic field contains forbidden markers."""

    if not isinstance(value, str):
        return_value = False
        return return_value
    return_value = bool(_FORBIDDEN_PATTERN.search(value))
    return return_value


def _mention_counter_kind(entity_kind: str) -> str:
    if entity_kind in ("bot", "user"):
        return_value = "user"
    elif entity_kind == "platform_role":
        return_value = "role"
    elif entity_kind == "channel":
        return_value = "channel"
    else:
        return_value = "entity"
    return return_value


def _legacy_placeholder(counter_kind: str, occurrence_index: int) -> str:
    if counter_kind == "channel":
        return_value = f"#mentioned-channel-{occurrence_index}"
    else:
        return_value = f"@mentioned-{counter_kind}-{occurrence_index}"
    return return_value


def _display_fallback(
    *,
    entity_kind: str,
) -> str:
    label = semantic_entity_fallback_label(
        entity_kind=entity_kind,
        mention_context=False,
    )
    return label


def _mention_text_token(
    *,
    entity_kind: str,
    display_label: str,
) -> str:
    if entity_kind == "channel":
        return_value = f"#{display_label}"
    else:
        return_value = f"@{display_label}"
    return return_value


def _row_mention_replacements(row: dict[str, Any]) -> dict[str, str]:
    mentions = row.get("mentions")
    if not isinstance(mentions, list):
        return_value: dict[str, str] = {}
        return return_value

    occurrence_counts = {
        "user": 0,
        "role": 0,
        "channel": 0,
        "entity": 0,
    }
    replacements: dict[str, str] = {}
    for mention in mentions:
        if not isinstance(mention, dict):
            continue
        entity_kind = str(mention.get("entity_kind") or "unknown").strip()
        counter_kind = _mention_counter_kind(entity_kind)
        occurrence_counts[counter_kind] += 1
        placeholder = _legacy_placeholder(
            counter_kind,
            occurrence_counts[counter_kind],
        )
        display_label = normalize_mention_display_label(
            mention.get("display_name"),
        )
        if semantic_field_is_dirty(display_label):
            display_label = ""
        if not display_label:
            display_label = _display_fallback(
                entity_kind=entity_kind,
            )
        if not display_label:
            continue
        replacements[placeholder] = _mention_text_token(
            entity_kind=entity_kind,
            display_label=display_label,
        )

    return replacements


def _replace_placeholders(
    text: str,
    replacements: dict[str, str],
) -> str:
    repaired = text
    for placeholder, replacement in replacements.items():
        repaired = repaired.replace(placeholder, replacement)
    return_value = repaired
    return return_value


def _fallback_kind_from_platform_label(kind: str) -> str:
    if kind == "role":
        return_value = "platform_role"
    elif kind in ("bot", "channel", "entity", "user"):
        return_value = kind
    else:
        return_value = "unknown"
    return return_value


def _replace_platform_qualified_fallback(
    match: re.Match[str],
) -> str:
    kind = _fallback_kind_from_platform_label(match.group("kind"))
    mention_context = bool(match.group("prefix"))
    replacement = semantic_entity_fallback_label(
        entity_kind=kind,
        mention_context=mention_context,
    )
    return replacement


def _repair_text_field(
    text: str,
    replacements: dict[str, str],
) -> tuple[str, bool]:
    replaced = _replace_placeholders(text, replacements)
    replaced = _PLATFORM_QUALIFIED_FALLBACK_PATTERN.sub(
        _replace_platform_qualified_fallback,
        replaced,
    )
    replaced = _LEGACY_PLACEHOLDER_PATTERN.sub(" ", replaced)
    repaired = normalize_dirty_body_text(replaced)
    return_value = (repaired, semantic_field_is_dirty(repaired))
    return return_value


def _fallback_for_row_user(row: dict[str, Any]) -> str:
    label = _display_fallback(
        entity_kind="user",
    )
    return label


def _repair_reply_context(
    row: dict[str, Any],
    replacements: dict[str, str],
) -> tuple[dict[str, Any] | None, list[str]]:
    reply_context = row.get("reply_context")
    if not isinstance(reply_context, dict):
        return_value = (None, [])
        return return_value

    repaired_context = dict(reply_context)
    ambiguous_fields: list[str] = []
    display_name = repaired_context.get("reply_to_display_name")
    if semantic_field_is_dirty(display_name):
        fallback = _display_fallback(
            entity_kind="user",
        )
        if fallback:
            repaired_context["reply_to_display_name"] = fallback
        else:
            ambiguous_fields.append("reply_context.reply_to_display_name")

    reply_excerpt = repaired_context.get("reply_excerpt")
    if isinstance(reply_excerpt, str) and semantic_field_is_dirty(reply_excerpt):
        repaired_excerpt, ambiguous = _repair_text_field(
            reply_excerpt,
            replacements,
        )
        if repaired_excerpt:
            repaired_context["reply_excerpt"] = repaired_excerpt
        else:
            repaired_context.pop("reply_excerpt", None)
        if ambiguous:
            ambiguous_fields.append("reply_context.reply_excerpt")

    if repaired_context == reply_context:
        return_value = (None, ambiguous_fields)
        return return_value

    return_value = (repaired_context, ambiguous_fields)
    return return_value


def repair_conversation_row(row: dict[str, Any]) -> ConversationRepair:
    """Build a deterministic repair for one conversation-history row."""

    replacements = _row_mention_replacements(row)
    set_fields: dict[str, Any] = {}
    ambiguous_fields: list[str] = []
    requires_reembedding = False

    display_name = row.get("display_name")
    if semantic_field_is_dirty(display_name):
        fallback = _fallback_for_row_user(row)
        if fallback:
            set_fields["display_name"] = fallback
        else:
            ambiguous_fields.append("display_name")

    body_text = row.get("body_text")
    if isinstance(body_text, str) and semantic_field_is_dirty(body_text):
        repaired_body, ambiguous = _repair_text_field(body_text, replacements)
        set_fields["body_text"] = repaired_body
        requires_reembedding = repaired_body != body_text
        if ambiguous:
            ambiguous_fields.append("body_text")

    mentions = row.get("mentions")
    if isinstance(mentions, list):
        repaired_mentions = []
        mentions_changed = False
        for mention in mentions:
            if not isinstance(mention, dict):
                repaired_mentions.append(mention)
                continue
            repaired_mention = dict(mention)
            mention_display_name = repaired_mention.get("display_name")
            if semantic_field_is_dirty(mention_display_name):
                fallback = _display_fallback(
                    entity_kind=str(repaired_mention.get("entity_kind") or "unknown"),
                )
                if fallback:
                    repaired_mention["display_name"] = fallback
                    mentions_changed = True
                else:
                    ambiguous_fields.append("mentions.display_name")
            repaired_mentions.append(repaired_mention)
        if mentions_changed:
            set_fields["mentions"] = repaired_mentions

    repaired_reply_context, reply_ambiguous = _repair_reply_context(
        row,
        replacements,
    )
    ambiguous_fields.extend(reply_ambiguous)
    if repaired_reply_context is not None:
        set_fields["reply_context"] = repaired_reply_context

    repairable = bool(set_fields) and not ambiguous_fields
    return_value = ConversationRepair(
        repairable=repairable,
        set_fields=set_fields,
        unset_fields=(),
        requires_reembedding=requires_reembedding,
        ambiguous_fields=ambiguous_fields,
    )
    return return_value


def _repair_profile_account(account: dict[str, Any]) -> str:
    fallback = _display_fallback(
        entity_kind="user",
    )
    return fallback


def _backup_path(backup_dir: Path, name: str) -> Path:
    timestamp = storage_utc_now_iso().replace(":", "").replace("+", "Z")
    return_value = backup_dir / f"{name}_{timestamp}.json"
    return return_value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_util.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


async def build_semantic_identity_report(
    *,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Build a bounded pollution report without mutating storage."""

    conversation_rows = await list_semantic_identity_conversation_rows(
        batch_size=batch_size,
    )
    profile_rows = await list_semantic_identity_user_profile_rows(
        batch_size=batch_size,
    )
    user_memory_rows = await list_semantic_identity_user_memory_units(
        batch_size=batch_size,
    )
    shared_memory_rows = await list_semantic_identity_shared_memory_units(
        batch_size=batch_size,
    )

    repair_results = [
        repair_conversation_row(row)
        for row in conversation_rows
    ]
    repairable_rows = sum(1 for result in repair_results if result.repairable)
    ambiguous_rows = sum(
        1
        for result in repair_results
        if result.ambiguous_fields
    )
    requires_reembedding = sum(
        1
        for result in repair_results
        if result.requires_reembedding
    )

    dirty_accounts = await count_semantic_identity_user_profile_accounts()
    dirty_conversations = await count_semantic_identity_conversation_rows()
    dirty_user_memory = await count_semantic_identity_user_memory_units()
    dirty_shared_memory = await count_semantic_identity_shared_memory_units()
    report = {
        "generated_at": storage_utc_now_iso(),
        "dry_run": dry_run,
        "conversation_history": {
            "dirty_rows": dirty_conversations,
            "repairable_rows": repairable_rows,
            "ambiguous_rows": ambiguous_rows,
            "requires_reembedding": requires_reembedding,
        },
        "user_profiles": {
            "dirty_accounts": dirty_accounts,
            "repairable_accounts": sum(
                1
                for profile in profile_rows
                for account in profile.get("platform_accounts", [])
                if (
                    isinstance(account, dict)
                    and semantic_field_is_dirty(account.get("display_name"))
                    and _repair_profile_account(account)
                )
            ),
            "ambiguous_accounts": sum(
                1
                for profile in profile_rows
                for account in profile.get("platform_accounts", [])
                if (
                    isinstance(account, dict)
                    and semantic_field_is_dirty(account.get("display_name"))
                    and not _repair_profile_account(account)
                )
            ),
        },
        "derived_memory": {
            "dirty_user_memory_units": dirty_user_memory,
            "dirty_shared_memory_units": dirty_shared_memory,
            "repairable_units": 0,
            "deactivate_units": dirty_user_memory + sum(
                1
                for row in shared_memory_rows
                if str(row.get("memory_unit_id") or "").strip()
            ),
            "manual_review_units": sum(
                1
                for row in shared_memory_rows
                if not str(row.get("memory_unit_id") or "").strip()
            ),
        },
    }
    return report


async def apply_semantic_identity_repair(
    *,
    dry_run: bool,
    batch_size: int,
    backup_dir: Path | None,
) -> dict[str, Any]:
    """Run dry-run or apply mode for deterministic identity repair."""

    report = await build_semantic_identity_report(
        dry_run=dry_run,
        batch_size=batch_size,
    )
    if dry_run:
        return report
    if backup_dir is None:
        raise ValueError("backup_dir is required for apply")

    conversation_rows = await list_semantic_identity_conversation_rows(
        batch_size=batch_size,
    )
    profile_rows = await list_semantic_identity_user_profile_rows(
        batch_size=batch_size,
    )
    user_memory_rows = await list_semantic_identity_user_memory_units(
        batch_size=batch_size,
    )
    shared_memory_rows = await list_semantic_identity_shared_memory_units(
        batch_size=batch_size,
    )
    backup_files = {
        "conversation_history": str(_backup_path(backup_dir, "conversation_history")),
        "user_profiles": str(_backup_path(backup_dir, "user_profiles")),
        "user_memory_units": str(_backup_path(backup_dir, "user_memory_units")),
        "memory": str(_backup_path(backup_dir, "memory")),
    }
    _write_json(
        Path(backup_files["conversation_history"]),
        {"records": conversation_rows},
    )
    _write_json(Path(backup_files["user_profiles"]), {"records": profile_rows})
    _write_json(
        Path(backup_files["user_memory_units"]),
        {"records": user_memory_rows},
    )
    _write_json(Path(backup_files["memory"]), {"records": shared_memory_rows})

    repaired_conversation_rows = 0
    reembedded_rows = 0
    for row in conversation_rows:
        repair = repair_conversation_row(row)
        if not repair.repairable:
            continue
        await update_conversation_history_row_for_semantic_identity_repair(
            row_id=row["_id"],
            set_fields=repair.set_fields,
            unset_fields=repair.unset_fields,
            recompute_embedding=repair.requires_reembedding,
        )
        repaired_conversation_rows += 1
        if repair.requires_reembedding:
            reembedded_rows += 1

    repaired_profile_accounts = 0
    for profile in profile_rows:
        global_user_id = str(profile.get("global_user_id") or "").strip()
        accounts = profile.get("platform_accounts")
        if not global_user_id or not isinstance(accounts, list):
            continue
        for account in accounts:
            if not isinstance(account, dict):
                continue
            if not semantic_field_is_dirty(account.get("display_name")):
                continue
            fallback = _repair_profile_account(account)
            if not fallback:
                continue
            result = await update_user_profile_platform_account_display_name(
                global_user_id=global_user_id,
                platform=str(account.get("platform") or ""),
                platform_user_id=str(account.get("platform_user_id") or ""),
                display_name=fallback,
            )
            repaired_profile_accounts += int(result["modified_count"])

    archived_user_memory_units = 0
    for row in user_memory_rows:
        unit_id = str(row.get("unit_id") or "").strip()
        if not unit_id:
            continue
        result = await archive_user_memory_unit_for_semantic_identity_repair(
            unit_id=unit_id,
            reason="semantic_identity_pollution",
            storage_timestamp_utc=storage_utc_now_iso(),
        )
        archived_user_memory_units += int(result["modified_count"])

    rejected_shared_memory_units = 0
    for row in shared_memory_rows:
        memory_unit_id = str(row.get("memory_unit_id") or "").strip()
        if not memory_unit_id:
            continue
        await reject_memory_unit(
            active_unit_id=memory_unit_id,
            reason="semantic_identity_pollution",
            storage_timestamp_utc=storage_utc_now_iso(),
        )
        rejected_shared_memory_units += 1

    report["dry_run"] = False
    report["apply"] = {
        "backup_files": backup_files,
        "repaired_conversation_rows": repaired_conversation_rows,
        "reembedded_conversation_rows": reembedded_rows,
        "repaired_profile_accounts": repaired_profile_accounts,
        "archived_user_memory_units": archived_user_memory_units,
        "rejected_shared_memory_units": rejected_shared_memory_units,
    }
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line parser for semantic identity repair."""

    parser = argparse.ArgumentParser(
        description="Repair semantic identity pollution in durable storage.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Report only.")
    mode.add_argument("--apply", action="store_true", help="Apply repairs.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Maximum rows to inspect per dirty collection.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DRY_RUN_OUTPUT,
        help="Destination JSON report path.",
    )
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=None,
        help="Required backup directory for --apply.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


async def main() -> None:
    """Run the semantic identity repair command."""

    configure_stdout()
    parser = build_arg_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    try:
        report = await apply_semantic_identity_repair(
            dry_run=not args.apply,
            batch_size=args.batch_size,
            backup_dir=args.backup_dir,
        )
        _write_json(args.output, report)
        print(json_util.dumps(report, ensure_ascii=False, indent=2))
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
