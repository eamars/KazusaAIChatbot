"""Inspect or recreate approved vector-search indexes."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

APPROVED_VECTOR_SEARCH_COLLECTIONS = ("conversation_history",)


async def inspect_vector_search_index(collection_name: str) -> dict[str, Any]:
    """Inspect one approved vector-search index."""

    from kazusa_ai_chatbot.db.script_operations import (  # noqa: PLC0415
        inspect_vector_search_index as inspect_index,
    )

    result = await inspect_index(collection_name)
    return result


async def apply_vector_search_index(
    collection_name: str,
    *,
    wait_ready: bool,
) -> dict[str, Any]:
    """Recreate one approved vector-search index."""

    from kazusa_ai_chatbot.db.script_operations import (  # noqa: PLC0415
        apply_vector_search_index as apply_index,
    )

    result = await apply_index(collection_name, wait_ready=wait_ready)
    return result


async def ensure_vector_search_indexes(
    *,
    collections: list[str],
    apply: bool,
    wait_ready: bool,
) -> dict[str, Any]:
    """Inspect and optionally recreate approved vector-search indexes."""

    index_results: list[dict[str, Any]] = []
    for collection_name in collections:
        inspection = await inspect_vector_search_index(collection_name)
        if apply and bool(inspection["requires_recreate"]):
            apply_result = await apply_vector_search_index(
                collection_name,
                wait_ready=wait_ready,
            )
            index_results.append(apply_result)
            continue
        index_results.append(inspection)

    result = {
        "apply": apply,
        "wait_ready": wait_ready,
        "indexes": index_results,
    }
    return result


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description="Inspect or recreate approved Atlas vector-search indexes.",
    )
    parser.add_argument(
        "--collections",
        nargs="+",
        default=["conversation_history"],
        choices=sorted(APPROVED_VECTOR_SEARCH_COLLECTIONS),
        help="Collections whose vector-search indexes should be checked.",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect indexes without applying changes.",
    )
    mode_group.add_argument(
        "--apply",
        action="store_true",
        help="Recreate indexes that require configured filter fields.",
    )
    parser.add_argument(
        "--wait-ready",
        action="store_true",
        help="Wait for recreated indexes to become queryable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON artifact path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable project INFO logs.",
    )
    return parser


def _write_output(output_path: Path, result: dict[str, Any]) -> None:
    """Write a readable JSON result artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected index operation and close DB handles."""

    from kazusa_ai_chatbot.db import close_db  # noqa: PLC0415

    try:
        result = await ensure_vector_search_indexes(
            collections=list(args.collections),
            apply=bool(args.apply),
            wait_ready=bool(args.wait_ready),
        )
    finally:
        await close_db()
    return result


def main() -> None:
    """CLI entry point."""

    from scripts._db_export import (  # noqa: PLC0415
        configure_logging,
        configure_stdout,
        load_project_env,
    )

    parser = _parser()
    args = parser.parse_args()
    configure_stdout()
    configure_logging(bool(args.verbose))
    load_project_env()

    result = asyncio.run(_main_async(args))
    if args.output is not None:
        _write_output(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
