"""Run the daily reflection-backed character identity growth pass."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import date, timedelta

from scripts._db_export import (
    configure_logging,
    configure_stdout,
    load_project_env,
)

from kazusa_ai_chatbot.character_identity_growth.runner import (
    run_reflection_identity_growth_pass,
)
from kazusa_ai_chatbot.db import close_db, db_bootstrap
from kazusa_ai_chatbot.reflection_cycle import repository
from kazusa_ai_chatbot.time_boundary import (
    local_time_context_from_storage_utc,
    storage_utc_now_iso,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate daily reflection evidence for character identity growth"
        ),
    )
    parser.add_argument(
        "--character-local-date",
        default="",
        help=(
            "Character-local YYYY-MM-DD to evaluate; defaults to the "
            "previous character-local date"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Allow a validated identity revision to be promoted",
    )
    parser.add_argument(
        "--enable-revision-writes",
        action="store_true",
        help="Required explicit write gate when --apply is selected",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def _selected_local_date(value: str) -> str:
    """Validate an explicit date or return the previous local date."""

    if value:
        return date.fromisoformat(value).isoformat()
    local_context = local_time_context_from_storage_utc(
        storage_utc_now_iso(),
    )
    current_local_date = date.fromisoformat(
        local_context["current_local_datetime"][:10],
    )
    return (current_local_date - timedelta(days=1)).isoformat()


def _validate_write_gate(args: argparse.Namespace) -> None:
    """Require both explicit apply flags for revision writes."""

    if args.apply and not args.enable_revision_writes:
        raise ValueError(
            "--apply requires --enable-revision-writes"
        )
    if args.enable_revision_writes and not args.apply:
        raise ValueError(
            "--enable-revision-writes is valid only with --apply"
        )


async def main() -> None:
    """Run one daily identity evaluation and print its sanitized result."""

    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()
    _validate_write_gate(args)
    character_local_date = _selected_local_date(
        args.character_local_date,
    )

    await db_bootstrap()
    try:
        daily_runs = await repository.daily_channel_runs(
            character_local_date=character_local_date,
        )
        source_run_ids = sorted({
            str(document.get("run_id", "")).strip()
            for document in daily_runs
            if str(document.get("run_id", "")).strip()
            if document.get("status") in {None, "succeeded"}
        })
        result = await run_reflection_identity_growth_pass(
            character_local_date=character_local_date,
            source_reflection_run_ids=source_run_ids,
            dry_run=not args.apply,
            enable_revision_writes=args.enable_revision_writes,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the asynchronous command."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
