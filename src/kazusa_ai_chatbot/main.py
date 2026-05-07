"""Command-line entry point for the Kazusa brain service."""

from __future__ import annotations

import argparse

import uvicorn


def _build_parser() -> argparse.ArgumentParser:
    """Build the brain-service command-line parser."""

    parser = argparse.ArgumentParser(description="Run the Kazusa brain service.")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host interface for uvicorn to bind.",
    )
    parser.add_argument(
        "--port",
        default=8000,
        type=int,
        help="TCP port for uvicorn to bind.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload for local development.",
    )
    return parser


def main() -> None:
    """Run the FastAPI brain service with uvicorn."""

    parser = _build_parser()
    args = parser.parse_args()
    uvicorn.run(
        "kazusa_ai_chatbot.service:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
