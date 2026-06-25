"""Apply or dry-run TTL expiry for existing logging rows."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Any

from kazusa_ai_chatbot.config import AUDIT_LOG_TTL_DAYS, DEBUG_LOG_TTL_DAYS
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.logging_retention import (
    expiry_from_now,
    expiry_from_storage_iso,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now


RETENTION_TARGETS = (
    ("event_log_events", "occurred_at", AUDIT_LOG_TTL_DAYS),
    ("event_log_snapshots", "generated_at", AUDIT_LOG_TTL_DAYS),
    ("llm_trace_runs", "started_at", DEBUG_LOG_TTL_DAYS),
    ("llm_trace_steps", "created_at", DEBUG_LOG_TTL_DAYS),
)


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when available."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Apply TTL expiry fields to logging collections.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    return parser


def _row_expiry(
    row: dict[str, Any],
    *,
    timestamp_field: str,
    ttl_days: int,
) -> datetime:
    """Compute expiry for one legacy logging row."""

    raw_timestamp = row.get(timestamp_field)
    if isinstance(raw_timestamp, datetime):
        if raw_timestamp.tzinfo is None:
            source = raw_timestamp.replace(tzinfo=timezone.utc)
        else:
            source = raw_timestamp.astimezone(timezone.utc)
        expires_at = source + timedelta(days=ttl_days)
        return expires_at
    if isinstance(raw_timestamp, str) and raw_timestamp.strip():
        return expiry_from_storage_iso(raw_timestamp, ttl_days=ttl_days)
    return expiry_from_now(ttl_days=ttl_days)


async def apply_logging_retention(
    *,
    dry_run: bool,
    batch_size: int,
) -> dict[str, Any]:
    """Assign or delete legacy logging rows according to TTL policy."""

    db = await get_db()
    now = storage_utc_now()
    collections: list[dict[str, Any]] = []
    for collection_name, timestamp_field, ttl_days in RETENTION_TARGETS:
        collection = db[collection_name]
        missing_filter = {
            "$or": [
                {"expires_at": {"$exists": False}},
                {"expires_at": None},
            ]
        }
        total_missing = await collection.count_documents(missing_filter)
        expired_ids = []
        update_rows = []
        scanned = 0
        cursor = collection.find(missing_filter).batch_size(batch_size)
        async for row in cursor:
            scanned += 1
            expires_at = _row_expiry(
                row,
                timestamp_field=timestamp_field,
                ttl_days=ttl_days,
            )
            if expires_at <= now:
                expired_ids.append(row["_id"])
            else:
                update_rows.append((row["_id"], expires_at))

        deleted = 0
        updated = 0
        if not dry_run:
            if expired_ids:
                delete_result = await collection.delete_many(
                    {"_id": {"$in": expired_ids}},
                )
                deleted = int(delete_result.deleted_count)
            for row_id, expires_at in update_rows:
                update_result = await collection.update_one(
                    {"_id": row_id},
                    {"$set": {"expires_at": expires_at}},
                )
                updated += int(update_result.modified_count)

        collections.append({
            "collection": collection_name,
            "timestamp_field": timestamp_field,
            "ttl_days": ttl_days,
            "missing_expires_at": total_missing,
            "scanned": scanned,
            "would_delete": len(expired_ids),
            "would_update": len(update_rows),
            "deleted": deleted,
            "updated": updated,
        })

    report = {
        "mode": "dry_run" if dry_run else "apply",
        "batch_size": batch_size,
        "collections": collections,
    }
    return report


async def main() -> None:
    """Run the retention maintenance CLI."""

    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        report = await apply_logging_retention(
            dry_run=bool(args.dry_run),
            batch_size=args.batch_size,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
