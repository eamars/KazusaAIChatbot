"""Export sanitized event-log aggregates for operator diagnostics.

Typical use:
    python -m scripts.export_event_log --hours 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.time_boundary import storage_utc_now, storage_utc_now_iso


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when the active stream supports it."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _default_output_path() -> Path:
    """Build the default event-log diagnostics export path."""

    timestamp_utc = storage_utc_now().strftime("%Y%m%dT%H%M%SZ")
    output_path = (
        Path("test_artifacts")
        / "diagnostics"
        / f"event_log_{timestamp_utc}.json"
    )
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the aggregate export."""

    parser = argparse.ArgumentParser(
        description="Export sanitized event-log aggregate diagnostics.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Lookback window in hours.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination JSON path.",
    )
    return parser


async def build_export_document(*, window_hours: int) -> dict[str, Any]:
    """Build the sanitized aggregate export document.

    Args:
        window_hours: Lookback window used by all aggregate builders.

    Returns:
        JSON-serializable aggregate diagnostics document.
    """

    runtime_status = await event_logging.build_runtime_status(
        window_hours=window_hours,
    )
    reflection_stats = await event_logging.build_reflection_stats(
        window_hours=window_hours,
    )
    self_cognition_stats = await event_logging.build_self_cognition_stats(
        window_hours=window_hours,
    )
    snapshot_result = await event_logging.write_analysis_snapshot(
        window_hours=window_hours,
    )
    export_document = {
        "generated_at": storage_utc_now_iso(),
        "window_hours": window_hours,
        "runtime_status": runtime_status,
        "reflection_stats": reflection_stats,
        "self_cognition_stats": self_cognition_stats,
        "snapshot_write": snapshot_result,
    }
    return export_document


def write_export_document(
    *,
    output_path: Path,
    export_document: dict[str, Any],
) -> None:
    """Write one aggregate export document to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(export_document, ensure_ascii=False, indent=2)
    output_path.write_text(rendered, encoding="utf-8")


async def export_event_log(
    *,
    window_hours: int,
    output_path: Path | None = None,
) -> Path:
    """Build and write a sanitized event-log aggregate export.

    Args:
        window_hours: Lookback window in hours.
        output_path: Optional destination path. Defaults to diagnostics output.

    Returns:
        Path where the export document was written.
    """

    destination = output_path or _default_output_path()
    export_document = await build_export_document(window_hours=window_hours)
    write_export_document(
        output_path=destination,
        export_document=export_document,
    )
    return destination


async def main() -> None:
    """Run the event-log export CLI."""

    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    try:
        output_path = await export_event_log(
            window_hours=args.hours,
            output_path=args.output,
        )
        print(f"wrote event-log diagnostics to {output_path}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
