"""Command-line entry point for the Kazusa control console."""

from __future__ import annotations

import argparse

import uvicorn

from control_console.settings import ControlConsoleSettings


CONTROL_CONSOLE_GRACEFUL_SHUTDOWN_SECONDS = 3


def _build_parser() -> argparse.ArgumentParser:
    """Build the control-console command-line parser."""

    parser = argparse.ArgumentParser(description="Run the Kazusa control console.")
    parser.add_argument("--host", default=None, help="Host interface to bind.")
    parser.add_argument("--port", default=None, type=int, help="TCP port to bind.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload for local development.",
    )
    return parser


def main() -> None:
    """Run the local control console with uvicorn."""

    parser = _build_parser()
    args = parser.parse_args()
    settings = ControlConsoleSettings.from_env()
    host = args.host or settings.host
    port = args.port or settings.port
    uvicorn.run(
        "control_console.app:create_app",
        host=host,
        port=port,
        reload=args.reload,
        factory=True,
        timeout_graceful_shutdown=CONTROL_CONSOLE_GRACEFUL_SHUTDOWN_SECONDS,
    )
