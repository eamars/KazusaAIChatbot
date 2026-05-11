"""Dry-run or apply document-role re-embedding for text-vector collections."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

APPROVED_TEXT_VECTOR_COLLECTIONS = (
    "conversation_history",
    "memory",
    "user_memory_units",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse operator arguments for text-vector re-embedding."""

    parser = argparse.ArgumentParser(
        description="Recompute text-vector embeddings for approved collections.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--collections",
        nargs="+",
        choices=APPROVED_TEXT_VECTOR_COLLECTIONS,
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be a positive integer")
    return args


def _write_output(output_path: Path, result: Mapping[str, Any]) -> None:
    """Write the operator artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    """Run the requested dry-run or apply operation and close DB handles."""

    from kazusa_ai_chatbot.db import close_db  # noqa: PLC0415
    from kazusa_ai_chatbot.db.script_operations import (  # noqa: PLC0415
        reembed_text_vector_embeddings,
    )

    try:
        result = await reembed_text_vector_embeddings(
            collection_names=list(args.collections),
            batch_size=int(args.batch_size),
            apply=bool(args.apply),
        )
    finally:
        await close_db()
    return result


def main() -> None:
    """Run re-embedding maintenance from the command line."""

    from scripts._db_export import (  # noqa: PLC0415
        configure_logging,
        configure_stdout,
        load_project_env,
    )

    configure_stdout()
    configure_logging(False)
    load_project_env()
    args = parse_args()
    result = asyncio.run(_main_async(args))
    if args.output:
        _write_output(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
