"""One-off cleanup for the internal monologue residue collection.

The destructive path requires explicit confirmation of the configured database
and the exact collection name. The script reads only document identifiers during
the preflight and drops the collection through the database maintenance
boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from scripts._db_export import configure_logging, configure_stdout

from kazusa_ai_chatbot.config import MONGODB_DB_NAME
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import (
    drop_legacy_rag_collections as drop_named_collections,
    export_collection_rows,
)

COLLECTION_NAME = "internal_monologue_residue_state"
PREFLIGHT_LIMIT = 100_000


def _build_parser() -> argparse.ArgumentParser:
    """Build the guarded cleanup command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Clear the internal monologue residue collection selected by .env."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Drop the collection after the read-only preflight.",
    )
    parser.add_argument(
        "--confirm-database",
        default="",
        help="Required with --execute; must match MONGODB_DB_NAME.",
    )
    parser.add_argument(
        "--confirm-collection",
        default="",
        help="Required with --execute; must match the fixed collection name.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON path for the preflight and cleanup report.",
    )
    return parser


async def _read_document_ids() -> list[dict[str, Any]]:
    """Read bounded document identifiers for the destructive preflight."""

    rows = await export_collection_rows(
        collection_name=COLLECTION_NAME,
        filter_doc={},
        projection={"_id": 1},
        sort_doc={},
        limit=PREFLIGHT_LIMIT,
    )
    return rows


def _write_report(output_path: Path, report: dict[str, object]) -> None:
    """Write a compact cleanup report without document contents."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


async def main() -> None:
    """Run the read-only preflight and optional exact-collection drop."""

    configure_stdout()
    configure_logging(False)
    parser = _build_parser()
    args = parser.parse_args()

    if args.execute and args.confirm_database != MONGODB_DB_NAME:
        parser.error(
            "--confirm-database must exactly match the configured database"
        )
    if args.execute and args.confirm_collection != COLLECTION_NAME:
        parser.error(
            "--confirm-collection must exactly match the fixed target"
        )

    try:
        rows_before = await _read_document_ids()
        if len(rows_before) >= PREFLIGHT_LIMIT:
            raise RuntimeError(
                "preflight reached its row limit; cleanup was not attempted"
            )

        dropped: list[str] = []
        rows_after: list[dict[str, Any]] | None = None
        if args.execute:
            dropped = await drop_named_collections((COLLECTION_NAME,))
            rows_after = await _read_document_ids()

        report: dict[str, object] = {
            "database_name": MONGODB_DB_NAME,
            "collection_name": COLLECTION_NAME,
            "documents_before": len(rows_before),
            "preflight_limit": PREFLIGHT_LIMIT,
            "execute": bool(args.execute),
            "dropped_collections": dropped,
            "documents_after": (
                len(rows_after) if rows_after is not None else None
            ),
        }
        if args.output:
            _write_report(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(main())
