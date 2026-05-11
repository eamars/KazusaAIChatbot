"""Run the global character-growth background pass from the command line."""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from scripts._db_export import configure_logging, configure_stdout, load_project_env

from kazusa_ai_chatbot.config import CHARACTER_TIME_ZONE
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.global_character_growth import run_global_character_growth_pass


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for global growth runs."""

    parser = argparse.ArgumentParser(
        description="Run global character growth from reflection-promoted memory."
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--dry-run", action="store_true")
    mode_group.add_argument("--apply", action="store_true")
    parser.add_argument("--enable-trait-writes", action="store_true")
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--character-local-date", default="")
    parser.add_argument("--verbose", action="store_true")
    return parser


async def main() -> None:
    """Run the selected global growth command."""

    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    if args.apply and not args.enable_trait_writes:
        parser.error("--apply requires --enable-trait-writes")
    if args.dry_run and args.enable_trait_writes:
        parser.error("--enable-trait-writes is only valid with --apply")

    try:
        result = await run_global_character_growth_pass(
            character_local_date=_date_or_previous(args.character_local_date),
            dry_run=args.dry_run,
            enable_trait_writes=args.apply and args.enable_trait_writes,
            limit=args.limit,
        )
        _print_result(result)
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


def _print_result(result: dict) -> None:
    """Print a compact operator summary."""

    print(f"run id: {result.get('run_id', '')}")
    print(f"status: {result.get('status', '')}")
    print(f"dry run: {result.get('dry_run', '')}")
    print(f"eligible memory cards: {result.get('eligible_memory_cards', 0)}")
    print(f"accepted candidates: {result.get('accepted_candidate_count', 0)}")
    print(f"rejected candidates: {result.get('rejected_candidate_count', 0)}")
    print(f"trait updates: {result.get('trait_update_count', 0)}")
    print(f"promoted traits: {result.get('promoted_trait_count', 0)}")
    print(f"shadow projection: {result.get('shadow_projection_count', 0)}")
    print(f"input quality: {result.get('input_quality_density', '')}")
    print(f"warnings: {result.get('warning_count', 0)}")


def _date_or_previous(value: str) -> str:
    """Return the supplied date or yesterday in character-local time."""

    if value:
        return value
    local_now = datetime.now(ZoneInfo(CHARACTER_TIME_ZONE))
    previous_date = local_now.date() - timedelta(days=1)
    return_value = previous_date.isoformat()
    return return_value


if __name__ == "__main__":
    async_main()
