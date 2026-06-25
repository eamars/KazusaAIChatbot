"""Export one LLM trace and its linked audit/conversation rows.

Typical use:
    python -m scripts.export_llm_trace --trace-id llmtrace_...
    python -m scripts.export_llm_trace --dialog-text "14:30了"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db import script_operations
from kazusa_ai_chatbot.time_boundary import storage_utc_now, storage_utc_now_iso


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when available."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _default_output_path(trace_id: str) -> Path:
    """Build the default trace export path."""

    timestamp_utc = storage_utc_now().strftime("%Y%m%dT%H%M%SZ")
    safe_trace_id = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in trace_id
    )
    output_path = (
        Path("test_artifacts")
        / "diagnostics"
        / f"llm_trace_{safe_trace_id}_{timestamp_utc}.json"
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(description="Export protected LLM trace rows.")
    parser.add_argument("--trace-id", default="", help="Trace id to export.")
    parser.add_argument(
        "--dialog-text",
        default="",
        help="Visible dialog text used to resolve a trace id.",
    )
    parser.add_argument(
        "--delivery-tracking-id",
        default="",
        help="Assistant delivery_tracking_id used to resolve a trace id.",
    )
    parser.add_argument(
        "--platform-message-id",
        default="",
        help="Conversation platform_message_id used to resolve a trace id.",
    )
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    return parser


async def resolve_trace_id(
    *,
    trace_id: str,
    dialog_text: str,
    delivery_tracking_id: str,
    platform_message_id: str,
) -> str:
    """Resolve a trace id from direct id or visible conversation metadata."""

    clean_trace_id = trace_id.strip()
    if clean_trace_id:
        return clean_trace_id

    filters: list[dict[str, Any]] = []
    if delivery_tracking_id.strip():
        filters.append({"delivery_tracking_id": delivery_tracking_id.strip()})
    if platform_message_id.strip():
        filters.append({"platform_message_id": platform_message_id.strip()})
    if dialog_text.strip():
        filters.append({
            "body_text": {
                "$regex": dialog_text.strip(),
                "$options": "i",
            }
        })

    for filter_doc in filters:
        rows = await script_operations.export_collection_rows(
            collection_name="conversation_history",
            filter_doc=filter_doc,
            projection={"_id": 0, "llm_trace_id": 1},
            sort_doc={"timestamp": -1},
            limit=1,
        )
        if not rows:
            continue
        row_trace_id = str(rows[0].get("llm_trace_id", "")).strip()
        if row_trace_id:
            return row_trace_id

    raise ValueError("could not resolve llm_trace_id")


async def build_trace_export(*, trace_id: str) -> dict[str, Any]:
    """Build the complete trace export document."""

    trace_filter = {"trace_id": trace_id}
    runs = await script_operations.export_collection_rows(
        collection_name="llm_trace_runs",
        filter_doc=trace_filter,
        projection={},
        sort_doc={"started_at": -1},
        limit=10,
    )
    steps = await script_operations.export_collection_rows(
        collection_name="llm_trace_steps",
        filter_doc=trace_filter,
        projection={},
        sort_doc={"sequence": 1, "created_at": 1},
        limit=500,
    )
    events = await script_operations.export_collection_rows(
        collection_name="event_log_events",
        filter_doc={
            "$or": [
                {"correlation_id": trace_id},
                {"labels.llm_trace_id": trace_id},
                {"refs": {"$elemMatch": {
                    "ref_type": "llm_trace",
                    "ref_id": trace_id,
                }}},
            ]
        },
        projection={},
        sort_doc={"occurred_at": 1},
        limit=500,
    )
    conversation_rows = await script_operations.export_collection_rows(
        collection_name="conversation_history",
        filter_doc={"llm_trace_id": trace_id},
        projection={"embedding": 0},
        sort_doc={"timestamp": 1},
        limit=100,
    )
    document = {
        "generated_at": storage_utc_now_iso(),
        "query": {"trace_id": trace_id},
        "llm_trace_runs": runs,
        "llm_trace_steps": steps,
        "event_log_events": events,
        "conversation_history": conversation_rows,
    }
    return document


def write_trace_export(
    *,
    output_path: Path,
    export_document: dict[str, Any],
) -> None:
    """Write a JSON trace export."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(export_document, ensure_ascii=False, indent=2, default=str)
    output_path.write_text(rendered, encoding="utf-8")


async def export_trace(
    *,
    trace_id: str,
    output_path: Path | None = None,
) -> Path:
    """Build and write a trace export."""

    export_document = await build_trace_export(trace_id=trace_id)
    destination = output_path or _default_output_path(trace_id)
    write_trace_export(output_path=destination, export_document=export_document)
    return destination


async def main() -> None:
    """Run the trace export CLI."""

    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        trace_id = await resolve_trace_id(
            trace_id=args.trace_id,
            dialog_text=args.dialog_text,
            delivery_tracking_id=args.delivery_tracking_id,
            platform_message_id=args.platform_message_id,
        )
        output_path = await export_trace(
            trace_id=trace_id,
            output_path=args.output,
        )
        print(f"wrote LLM trace export to {output_path}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
