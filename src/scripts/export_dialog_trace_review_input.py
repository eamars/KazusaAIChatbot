"""Export compact LLM trace review input for a visible dialog."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.time_boundary import storage_utc_now
from scripts.export_llm_trace import build_trace_export, resolve_trace_id


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when available."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _default_output_path(trace_id: str) -> Path:
    """Build the default review export path."""

    timestamp_utc = storage_utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        Path("test_artifacts")
        / "llm_debug"
        / f"dialog_trace_review_{trace_id}_{timestamp_utc}.json"
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(
        description="Export compact dialog trace review input.",
    )
    parser.add_argument("--trace-id", default="", help="Trace id to export.")
    parser.add_argument("--dialog-text", default="", help="Visible dialog text.")
    parser.add_argument("--delivery-tracking-id", default="")
    parser.add_argument("--platform-message-id", default="")
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    return parser


def build_review_input(trace_export: dict[str, Any]) -> dict[str, Any]:
    """Build a compact review document from the full trace export."""

    steps = trace_export["llm_trace_steps"]
    review_steps = [
        {
            "sequence": step.get("sequence", 0),
            "stage_name": step.get("stage_name", ""),
            "route_name": step.get("route_name", ""),
            "model_name": step.get("model_name", ""),
            "status": step.get("status", ""),
            "parse_status": step.get("parse_status", ""),
            "prompt_chars": step.get("prompt_chars", 0),
            "output_chars": step.get("output_chars", 0),
            "output_state_fields": step.get("output_state_fields", []),
            "raw_messages": step.get("raw_messages", []),
            "raw_response_text": step.get("raw_response_text", ""),
            "parsed_output": step.get("parsed_output", {}),
        }
        for step in steps
    ]
    document = {
        "trace_id": trace_export["query"]["trace_id"],
        "conversation_history": trace_export["conversation_history"],
        "event_log_events": trace_export["event_log_events"],
        "llm_trace_steps": review_steps,
    }
    return document


def write_review_input(
    *,
    output_path: Path,
    document: dict[str, Any],
) -> None:
    """Write the review input JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


async def main() -> None:
    """Run the review export CLI."""

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
        trace_export = await build_trace_export(trace_id=trace_id)
        document = build_review_input(trace_export)
        output_path = args.output or _default_output_path(trace_id)
        write_review_input(output_path=output_path, document=document)
        print(f"wrote dialog trace review input to {output_path}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
