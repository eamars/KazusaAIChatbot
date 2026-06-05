"""CLI and environment parsing for the NapCat QQ adapter."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from kazusa_ai_chatbot.logging_config import configure_adapter_logging


configure_adapter_logging()
logger = logging.getLogger(__name__)


def main() -> None:
    """Start the NapCat QQ websocket adapter from command-line arguments."""

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="NapCat QQ adapter for Kazusa Brain Service",
    )
    parser.add_argument(
        "--channels",
        type=str,
        nargs="*",
        default=None,
        help=(
            "QQ Group IDs to actively participate in. "
            "Other groups will be listen-only."
        ),
    )
    parser.add_argument(
        "--runtime-host",
        type=str,
        default=os.getenv("ADAPTER_RUNTIME_HOST", "127.0.0.1"),
    )
    parser.add_argument(
        "--runtime-port",
        type=int,
        default=int(os.getenv("NAPCAT_RUNTIME_PORT", "8011")),
    )
    parser.add_argument(
        "--runtime-public-url",
        type=str,
        default=os.getenv("ADAPTER_RUNTIME_PUBLIC_URL", ""),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=float(os.getenv("ADAPTER_HEARTBEAT_SECONDS", "30")),
    )

    args = parser.parse_args()

    ws_url = os.getenv("NAPCAT_WS_URL")
    if not ws_url:
        logger.error("NAPCAT_WS_URL environment variable is required")
        sys.exit(1)

    ws_token = os.getenv("NAPCAT_WS_TOKEN")
    if not ws_token:
        logger.error("NAPCAT_WS_TOKEN environment variable is required")
        sys.exit(1)

    brain_url = os.getenv("BRAIN_URL")
    if not brain_url:
        logger.error("BRAIN_URL environment variable is required")
        sys.exit(1)

    brain_response_timeout = int(os.getenv("BRAIN_RESPONSE_TIMEOUT", "120"))
    runtime_public_url = (
        args.runtime_public_url or f"http://127.0.0.1:{args.runtime_port}"
    )
    runtime_shared_secret = os.getenv("ADAPTER_RUNTIME_SHARED_SECRET", "")

    from .ws_adapter import NapCatWSAdapter

    adapter = NapCatWSAdapter(
        ws_url=ws_url,
        ws_token=ws_token,
        brain_url=brain_url,
        brain_response_timeout=brain_response_timeout,
        runtime_host=args.runtime_host,
        runtime_port=args.runtime_port,
        runtime_public_url=runtime_public_url,
        runtime_shared_secret=runtime_shared_secret,
        heartbeat_seconds=args.heartbeat_seconds,
        channel_ids=args.channels,
        debug_modes={},
    )

    try:
        asyncio.run(adapter.connect())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        asyncio.run(adapter.close())
