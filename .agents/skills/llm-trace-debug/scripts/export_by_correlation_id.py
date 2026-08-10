"""Export runtime events and protected LLM traces for one chat correlation id."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import sys
from typing import Any

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db import script_operations
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from scripts._db_export import default_output_path
from scripts.export_llm_trace import build_trace_export, write_trace_export


MAX_CORRELATION_EVENT_ROWS = 500
MAX_CONVERSATION_ROWS = 100


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when the active stream supports it."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the correlation-export command-line parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Export exact runtime events and linked protected LLM traces for "
            "one chat correlation id."
        ),
    )
    parser.add_argument(
        "correlation_id",
        help="Identifier in chat:<platform>:<channel-ref>:<message-id> form.",
    )
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    return parser


def parse_chat_correlation_id(correlation_id: str) -> tuple[str, str, str]:
    """Parse the stable chat correlation identity.

    Args:
        correlation_id: Full service-generated chat correlation identifier.

    Returns:
        Platform, hashed channel reference, and platform message identifier.

    Raises:
        ValueError: If the identifier does not match the chat correlation
            contract.
    """

    clean_id = correlation_id.strip()
    pieces = clean_id.split(":", maxsplit=3)
    if (
        len(pieces) != 4
        or pieces[0] != "chat"
        or any(not piece for piece in pieces[1:])
    ):
        raise ValueError(
            "correlation id must use "
            "chat:<platform>:<channel-ref>:<message-id>"
        )
    _, platform, channel_ref, platform_message_id = pieces
    parsed_identity = platform, channel_ref, platform_message_id
    return parsed_identity


async def build_correlation_export(
    correlation_id: str,
) -> dict[str, Any]:
    """Join exact runtime events to conversation rows and protected traces.

    Args:
        correlation_id: Full service-generated chat correlation identifier.

    Returns:
        One export document containing correlation events, conversation rows,
        resolved trace identifiers, and complete protected trace exports.
    """

    platform, channel_ref, platform_message_id = parse_chat_correlation_id(
        correlation_id,
    )
    correlation_events = await script_operations.export_collection_rows(
        collection_name="event_log_events",
        filter_doc={"correlation_id": correlation_id.strip()},
        projection={},
        sort_doc={"occurred_at": 1},
        limit=MAX_CORRELATION_EVENT_ROWS,
    )
    conversation_rows = await script_operations.export_collection_rows(
        collection_name="conversation_history",
        filter_doc={
            "platform": platform,
            "platform_message_id": platform_message_id,
        },
        projection={"embedding": 0},
        sort_doc={"timestamp": 1},
        limit=MAX_CONVERSATION_ROWS,
    )

    trace_ids: list[str] = []
    for row in conversation_rows:
        trace_id = str(row.get("llm_trace_id", "")).strip()
        if trace_id and trace_id not in trace_ids:
            trace_ids.append(trace_id)

    trace_exports: list[dict[str, Any]] = []
    for trace_id in trace_ids:
        trace_export = await build_trace_export(trace_id=trace_id)
        trace_exports.append(trace_export)

    export_document = {
        "generated_at": storage_utc_now_iso(),
        "query": {
            "correlation_id": correlation_id.strip(),
            "platform": platform,
            "channel_ref": channel_ref,
            "platform_message_id": platform_message_id,
        },
        "correlation_event_log_events": correlation_events,
        "conversation_history": conversation_rows,
        "resolved_llm_trace_ids": trace_ids,
        "llm_traces": trace_exports,
    }
    return export_document


async def main() -> None:
    """Run the correlation-id trace exporter."""

    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        export_document = await build_correlation_export(args.correlation_id)
        output_path = args.output or default_output_path(
            "llm_trace_correlation",
            args.correlation_id,
        )
        write_trace_export(
            output_path=output_path,
            export_document=export_document,
        )
        print(f"wrote correlation trace export to {output_path}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async exporter."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
