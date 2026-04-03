"""Entry point for the Role-Play Discord Bot.

Usage:
    python main.py --personality personalities/example.json
    python main.py --personality personalities/example.json --channels 123456789 987654321
    python main.py --personality personalities/example.json --no-listen-all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the src directory is on the import path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from kazusa_ai_chatbot.discord_bot import run_bot


def main():
    parser = argparse.ArgumentParser(description="Role-Play Discord Bot")
    parser.add_argument(
        "--personality",
        type=str,
        required=True,
        help="Path to the personality JSON file",
    )
    parser.add_argument(
        "--channels",
        type=int,
        nargs="*",
        default=None,
        help="Discord channel IDs to listen in (omit to require @mention)",
    )
    parser.add_argument(
        "--no-listen-all",
        action="store_true",
        default=False,
        help="Disable listening in all channels (respond to @mentions only)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    # Suppress noisy MCP ping-validation warnings (harmless JSON-RPC pings
    # that the mcp client doesn't recognise as valid ServerNotification).
    logging.getLogger("mcp.client.session").setLevel(logging.ERROR)

    personality_path = Path(args.personality)
    if not personality_path.exists():
        logging.error("Personality file not found: %s", personality_path)
        sys.exit(1)

    listen_all = not args.no_listen_all and args.channels is None

    logging.info("Starting bot with personality: %s", personality_path)
    if listen_all:
        logging.info("Listening in ALL channels")
    elif args.channels:
        logging.info("Listening in channels: %s", args.channels)
    else:
        logging.info("No channels specified — bot will respond to @mentions only")

    run_bot(
        personality_path=personality_path,
        channel_ids=args.channels,
        listen_all=listen_all,
    )


if __name__ == "__main__":
    main()
