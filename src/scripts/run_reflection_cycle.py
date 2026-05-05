"""Run production reflection-cycle workers from the command line."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
from datetime import timedelta
from zoneinfo import ZoneInfo

from scripts._db_export import configure_logging, configure_stdout, load_project_env

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.config import CHARACTER_TIME_ZONE
from kazusa_ai_chatbot.reflection_cycle import (
    run_daily_channel_reflection_cycle,
    run_global_reflection_promotion,
    run_hourly_reflection_cycle,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for reflection worker runs."""

    parser = argparse.ArgumentParser(
        description="Run production reflection-cycle hourly, daily, or promotion work."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    hourly = subparsers.add_parser("hourly")
    hourly.add_argument("--dry-run", action="store_true")
    hourly.add_argument("--now", help="Optional ISO timestamp for the run clock.")

    daily = subparsers.add_parser("daily")
    daily.add_argument("--dry-run", action="store_true")
    daily.add_argument("--character-local-date", default="")

    promote = subparsers.add_parser("promote")
    promote.add_argument("--dry-run", action="store_true")
    promote.add_argument("--character-local-date", default="")
    promote.add_argument("--enable-memory-writes", action="store_true")

    parser.add_argument("--verbose", action="store_true")
    return parser


async def main() -> None:
    """Run the selected reflection worker command."""

    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    try:
        if args.command == "hourly":
            result = await run_hourly_reflection_cycle(
                now=_parse_optional_datetime(args.now),
                dry_run=args.dry_run,
            )
        elif args.command == "daily":
            result = await run_daily_channel_reflection_cycle(
                character_local_date=_date_or_previous(args.character_local_date),
                dry_run=args.dry_run,
            )
        else:
            result = await run_global_reflection_promotion(
                character_local_date=_date_or_previous(args.character_local_date),
                dry_run=args.dry_run,
                enable_memory_writes=args.enable_memory_writes,
            )
        print(f"run kind: {result.run_kind}")
        print(f"dry run: {result.dry_run}")
        print(f"processed: {result.processed_count}")
        print(f"succeeded: {result.succeeded_count}")
        print(f"failed: {result.failed_count}")
        print(f"skipped: {result.skipped_count}")
        print(f"deferred: {result.deferred}")
        if result.defer_reason:
            print(f"defer reason: {result.defer_reason}")
        print(f"run ids: {', '.join(result.run_ids)}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


def _date_or_previous(value: str) -> str:
    """Return the supplied date or yesterday in character-local time."""

    if value:
        return value
    local_now = datetime.now(ZoneInfo(CHARACTER_TIME_ZONE))
    previous_date = local_now.date() - timedelta(days=1)
    return_value = previous_date.isoformat()
    return return_value


def _parse_optional_datetime(value: str | None) -> datetime | None:
    """Parse an optional ISO timestamp argument."""

    if not value:
        return_value = None
        return return_value
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    return parsed


if __name__ == "__main__":
    async_main()
