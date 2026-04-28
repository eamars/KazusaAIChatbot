"""Export recent conversation history from MongoDB to JSON.

Typical use:
    python -m scripts.export_chat_history 673225019 --platform qq --hours 4
    python -m scripts.export_chat_history 673225019 --platform qq --limit 30
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts._db_export import (
    DEFAULT_EXCLUDED_FIELDS,
    configure_logging,
    configure_stdout,
    default_output_path,
    load_project_env,
    timestamp_hours_ago,
    utc_now,
    write_json_export,
)
from kazusa_ai_chatbot.db import close_db, get_conversation_history


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured argument parser for chat-history export.
    """
    parser = argparse.ArgumentParser(description="Export conversation_history rows for one channel.")
    parser.add_argument("platform_channel_id", help="Platform channel/group/private id to export.")
    parser.add_argument("--platform", default="qq", help="Platform filter, for example qq or discord.")
    parser.add_argument("--hours", type=float, help="Only export rows from the last N hours.")
    parser.add_argument("--from-timestamp", help="Inclusive ISO-8601 timestamp lower bound.")
    parser.add_argument("--to-timestamp", help="Inclusive ISO-8601 timestamp upper bound.")
    parser.add_argument("--limit", type=int, default=500, help="Maximum rows to export.")
    parser.add_argument("--include-embeddings", action="store_true", help="Include vector embeddings in output.")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show project database logs.")
    return parser


async def main() -> None:
    """Run the chat-history export CLI.

    Returns:
        None.
    """
    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    now = utc_now()
    from_timestamp = args.from_timestamp
    if args.hours is not None:
        from_timestamp = timestamp_hours_ago(args.hours, now=now)
    to_timestamp = args.to_timestamp or now.isoformat()
    exclude_fields = [] if args.include_embeddings else list(DEFAULT_EXCLUDED_FIELDS)
    output_path = args.output or default_output_path("chat_history", args.platform_channel_id)

    try:
        records = await get_conversation_history(
            platform=args.platform,
            platform_channel_id=args.platform_channel_id,
            limit=args.limit,
            from_timestamp=from_timestamp,
            to_timestamp=to_timestamp,
        )
        query = {
            "collection": "conversation_history",
            "platform": args.platform,
            "platform_channel_id": args.platform_channel_id,
            "from_timestamp": from_timestamp,
            "to_timestamp": to_timestamp,
            "limit": args.limit,
            "sort": "timestamp ascending after selecting latest matching rows",
        }
        write_json_export(
            output_path=output_path,
            query=query,
            records_key="messages",
            records=[dict(record) for record in records],
            exclude_fields=exclude_fields,
        )
        print(f"wrote {len(records)} messages to {output_path}")
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
