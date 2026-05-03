"""Inject a single world-knowledge entry into the database.

Stores entries in the ``memory`` collection.

Usage examples::

    python scripts/inject_knowledge.py \
        --name "Eiffel Tower height" \
        --content "The Eiffel Tower in Paris stands 330 metres tall including its antenna."
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timezone

from kazusa_ai_chatbot.db._client import close_db
from kazusa_ai_chatbot.db.memory import save_memory
from kazusa_ai_chatbot.db.schemas import MemoryDoc

_VALID_MEMORY_TYPES = ("fact", "narrative", "impression", "defense_rule")
_DEFAULT_MEMORY_TYPE = "fact"
_DEFAULT_SOURCE_KIND = "external_imported"

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """Return the configured argument parser."""
    parser = argparse.ArgumentParser(
        prog="inject_knowledge",
        description="Inject a single world-knowledge entry into MongoDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--name", required=True, help="Human-readable label for this knowledge entry.")
    parser.add_argument("--content", required=True, help="The knowledge text to store.")
    parser.add_argument(
        "--type",
        dest="memory_type",
        choices=_VALID_MEMORY_TYPES,
        default=_DEFAULT_MEMORY_TYPE,
        help=f"Memory type for the ``memory`` collection. Default: {_DEFAULT_MEMORY_TYPE}.",
    )
    parser.add_argument(
        "--note",
        dest="confidence_note",
        default="",
        help="Optional free-form confidence or source note.",
    )

    return parser


async def _inject_memory(args: argparse.Namespace, timestamp: str) -> None:
    """Store a global world-knowledge fact in the ``memory`` collection.

    Args:
        args: Parsed CLI arguments.
        timestamp: ISO-8601 UTC timestamp for the new document.
    """
    doc: MemoryDoc = {
        "memory_name": args.name,
        "content": args.content,
        "source_global_user_id": "",
        "memory_type": args.memory_type,
        "source_kind": _DEFAULT_SOURCE_KIND,
        "confidence_note": args.confidence_note,
        "status": "active",
        "expiry_timestamp": None,
    }

    try:
        await save_memory(doc, timestamp)
    except Exception as exc:
        logger.exception(f"Failed to save to memory collection: {exc}")
        raise RuntimeError(f"Failed to save to memory collection: {exc}") from exc

    print(f"[memory] Saved: '{args.name}'")


async def main(args: argparse.Namespace) -> None:
    """Orchestrate the injection based on parsed arguments.

    Args:
        args: Parsed CLI arguments.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    try:
        await _inject_memory(args, timestamp)
    finally:
        await close_db()


if __name__ == "__main__":
    parser = _build_parser()
    parsed = parser.parse_args()

    try:
        asyncio.run(main(parsed))
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
