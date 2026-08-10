"""Count or clear the two approved background-task history collections."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from typing import Any

from kazusa_ai_chatbot.accepted_task.models import ACCEPTED_TASKS_COLLECTION
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_JOBS_COLLECTION,
)
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db._client import get_db


CONFIRMATION_PHRASE = "DELETE_BACKGROUND_WORK_JOBS_AND_ACCEPTED_TASKS"
TARGET_COLLECTIONS = (
    BACKGROUND_WORK_JOBS_COLLECTION,
    ACCEPTED_TASKS_COLLECTION,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the reviewed cutover command-line controls."""

    parser = argparse.ArgumentParser(
        description=(
            "Count or clear the approved background-task history collections."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Delete every row from the two approved collections.",
    )
    parser.add_argument(
        "--confirm",
        default="",
        help="Required exact confirmation phrase when --execute is supplied.",
    )
    args = parser.parse_args(argv)
    return args


async def clear_background_task_history(*, execute: bool) -> dict[str, object]:
    """Count the exact allowlist and optionally delete its rows."""

    db = await get_db()
    before = await _collection_counts(db)
    deleted = {collection_name: 0 for collection_name in TARGET_COLLECTIONS}
    if execute:
        for collection_name in TARGET_COLLECTIONS:
            delete_result = await db[collection_name].delete_many({})
            deleted[collection_name] = int(delete_result.deleted_count)
    remaining = await _collection_counts(db)
    report = {
        "mode": "execute" if execute else "dry_run",
        "before": before,
        "deleted": deleted,
        "remaining": remaining,
    }
    if execute and any(remaining.values()):
        raise RuntimeError("background-task history verification found rows")
    return report


async def _collection_counts(db: Any) -> dict[str, int]:
    """Return counts for the exact destructive allowlist only."""

    counts: dict[str, int] = {}
    for collection_name in TARGET_COLLECTIONS:
        counts[collection_name] = int(
            await db[collection_name].count_documents({}),
        )
    return counts


async def _run(args: argparse.Namespace) -> dict[str, object]:
    """Validate destructive controls and execute one reviewed operation."""

    if args.execute and args.confirm != CONFIRMATION_PHRASE:
        raise ValueError("--execute requires the exact --confirm phrase")
    return await clear_background_task_history(execute=bool(args.execute))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the maintenance boundary and emit only count summaries."""

    args = parse_args(argv)
    try:
        report = asyncio.run(_run(args))
    except (RuntimeError, ValueError):
        print("Background-task history operation failed.", file=sys.stderr)
        return 1
    except Exception:
        print("Background-task history database operation failed.", file=sys.stderr)
        return 1
    finally:
        asyncio.run(close_db())
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
