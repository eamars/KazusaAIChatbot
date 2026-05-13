"""CLI helper to print recent event-log operator status.

Typical use:
    python -m scripts.fetch_ops_status
    python -m scripts.fetch_ops_status 6 --json
    fetch-ops-status --hours 24
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from kazusa_ai_chatbot import event_logging
from kazusa_ai_chatbot.config import SELF_COGNITION_ENABLED
from kazusa_ai_chatbot.db import close_db

DEFAULT_WINDOW_HOURS = 24


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when the active stream supports it."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for recent ops status lookup.

    Returns:
        Configured parser for the read-only ops status command.
    """

    parser = argparse.ArgumentParser(
        description="Fetch recent sanitized event-log operator status.",
    )
    parser.add_argument(
        "lookback_hours",
        nargs="?",
        type=int,
        help="Lookback window in hours. Defaults to 24.",
    )
    parser.add_argument(
        "--hours",
        type=int,
        help="Lookback window in hours. Use instead of positional hours.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the aggregate status document as JSON.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show project database logs while running the lookup.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Keep the status lookup quiet unless verbose output is requested.

    Args:
        verbose: Whether to preserve project INFO logs.
    """

    if not verbose:
        logging.getLogger("kazusa_ai_chatbot").setLevel(logging.WARNING)


def _resolve_window_hours(
    *,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> int:
    """Resolve and validate the lookback window from CLI arguments.

    Args:
        args: Parsed command-line namespace.
        parser: Parser used to report invalid argument combinations.

    Returns:
        Non-negative lookback window in hours.
    """

    if args.lookback_hours is not None and args.hours is not None:
        parser.error("pass either positional hours or --hours, not both")

    window_hours = args.hours
    if window_hours is None:
        window_hours = args.lookback_hours
    if window_hours is None:
        window_hours = DEFAULT_WINDOW_HOURS
    if window_hours < 0:
        parser.error("hours must be non-negative")

    return window_hours


def _compact_json(value: Any) -> str:
    """Serialize a value as stable, readable JSON.

    Args:
        value: JSON-compatible aggregate status document.

    Returns:
        Pretty-printed JSON with non-ASCII text preserved.
    """

    return_value = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return return_value


def _safe_mapping(value: object) -> dict[str, Any]:
    """Return a mapping value or an empty mapping for tolerant formatting.

    Args:
        value: Status section returned by an aggregate builder.

    Returns:
        The original mapping when it is a dictionary, otherwise an empty one.
    """

    if isinstance(value, dict):
        safe_value = value
    else:
        safe_value = {}
    return safe_value


def _display_value(value: object, *, fallback: str) -> str:
    """Render optional aggregate values with a stable fallback label.

    Args:
        value: Value from an aggregate ops status payload.
        fallback: Label to use when the value is empty.

    Returns:
        Stripped display string or the fallback label.
    """

    display_text = str(value or "").strip()
    if not display_text:
        display_text = fallback
    return display_text


def _format_latest_event(latest: dict[str, Any]) -> str:
    """Format latest-event refs without exposing raw event payloads.

    Args:
        latest: Aggregate ``latest`` section from an ops status builder.

    Returns:
        Compact latest-event summary for terminal output.
    """

    event_id = _display_value(latest.get("event_id"), fallback="none")
    run_id = _display_value(latest.get("run_id"), fallback="none")
    occurred_at = _display_value(latest.get("occurred_at"), fallback="none")
    status = _display_value(latest.get("status"), fallback="unknown")
    latest_text = (
        f"event_id={event_id} run_id={run_id} "
        f"status={status} occurred_at={occurred_at}"
    )
    return latest_text


async def build_ops_status_document(*, window_hours: int) -> dict[str, Any]:
    """Build the read-only recent ops status document.

    Args:
        window_hours: Lookback window passed to all event-log aggregate builders.

    Returns:
        JSON-serializable document with aggregate ops status sections.
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
    status_document = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_hours": window_hours,
        "runtime_status": runtime_status,
        "reflection_stats": reflection_stats,
        "self_cognition_stats": self_cognition_stats,
        "self_cognition_runtime": {
            "enabled": SELF_COGNITION_ENABLED,
        },
    }
    return status_document


def format_ops_status_document(status_document: dict[str, Any]) -> str:
    """Format a recent ops status document for terminal inspection.

    Args:
        status_document: Aggregate document returned by
            ``build_ops_status_document``.

    Returns:
        Human-readable status summary with aggregate counts and refs only.
    """

    runtime_status = _safe_mapping(status_document.get("runtime_status"))
    reflection_stats = _safe_mapping(status_document.get("reflection_stats"))
    self_cognition_stats = _safe_mapping(
        status_document.get("self_cognition_stats"),
    )
    self_cognition_runtime = _safe_mapping(
        status_document.get("self_cognition_runtime"),
    )

    process = _safe_mapping(runtime_status.get("process"))
    workers = _safe_mapping(runtime_status.get("workers"))
    reflection_worker = _safe_mapping(workers.get("reflection_cycle"))
    self_cognition_worker = _safe_mapping(workers.get("self_cognition"))
    runtime_descriptors = _safe_mapping(
        runtime_status.get("semantic_descriptors"),
    )

    reflection_counts = _safe_mapping(reflection_stats.get("counts"))
    reflection_latest = _safe_mapping(reflection_stats.get("latest"))
    reflection_descriptors = _safe_mapping(
        reflection_stats.get("semantic_descriptors"),
    )

    self_cognition_counts = _safe_mapping(self_cognition_stats.get("counts"))
    self_cognition_latest = _safe_mapping(self_cognition_stats.get("latest"))
    self_cognition_descriptors = _safe_mapping(
        self_cognition_stats.get("semantic_descriptors"),
    )

    generated_at = _display_value(
        status_document.get("generated_at"),
        fallback="unknown",
    )
    window_hours = _display_value(
        status_document.get("window_hours"),
        fallback="unknown",
    )
    process_status = _display_value(
        process.get("last_status"),
        fallback="unknown",
    )
    process_event_at = _display_value(
        process.get("last_event_at"),
        fallback="none",
    )
    worker_error_level = _display_value(
        runtime_descriptors.get("worker_error_level"),
        fallback="unknown",
    )
    reflection_worker_status = _display_value(
        reflection_worker.get("last_status"),
        fallback="unknown",
    )
    reflection_worker_event_at = _display_value(
        reflection_worker.get("last_event_at"),
        fallback="none",
    )
    self_cognition_worker_status = _display_value(
        self_cognition_worker.get("last_status"),
        fallback="unknown",
    )
    self_cognition_worker_event_at = _display_value(
        self_cognition_worker.get("last_event_at"),
        fallback="none",
    )
    reflection_succeeded = _display_value(
        reflection_counts.get("succeeded"),
        fallback="0",
    )
    reflection_failed = _display_value(
        reflection_counts.get("failed"),
        fallback="0",
    )
    reflection_skipped = _display_value(
        reflection_counts.get("skipped"),
        fallback="0",
    )
    reflection_health = _display_value(
        reflection_descriptors.get("reflection_health"),
        fallback="unknown",
    )
    self_cognition_runs = _display_value(
        self_cognition_counts.get("runs"),
        fallback="0",
    )
    self_cognition_dispatch_accepted = _display_value(
        self_cognition_counts.get("dispatch_accepted"),
        fallback="0",
    )
    self_cognition_liveness = _display_value(
        self_cognition_descriptors.get("self_cognition_liveness"),
        fallback="unknown",
    )
    self_cognition_enabled = _display_value(
        self_cognition_runtime.get("enabled"),
        fallback="unknown",
    )

    lines = [
        f"ops_status_generated_at: {generated_at}",
        f"window_hours: {window_hours}",
        "",
        "runtime:",
        f"  process_status: {process_status}",
        f"  process_last_event_at: {process_event_at}",
        f"  worker_error_level: {worker_error_level}",
        "  workers:",
        (
            "    reflection_cycle: "
            f"{reflection_worker_status} at {reflection_worker_event_at}"
        ),
        (
            "    self_cognition: "
            f"{self_cognition_worker_status} at {self_cognition_worker_event_at}"
        ),
        "",
        "reflection:",
        f"  succeeded: {reflection_succeeded}",
        f"  failed: {reflection_failed}",
        f"  skipped: {reflection_skipped}",
        f"  health: {reflection_health}",
        f"  latest: {_format_latest_event(reflection_latest)}",
        "",
        "self_cognition:",
        f"  enabled: {self_cognition_enabled}",
        f"  runs: {self_cognition_runs}",
        f"  dispatch_accepted: {self_cognition_dispatch_accepted}",
        f"  liveness: {self_cognition_liveness}",
        f"  latest: {_format_latest_event(self_cognition_latest)}",
    ]
    formatted = "\n".join(lines)
    return formatted


async def main() -> None:
    """Run the recent ops status CLI."""

    _configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    _configure_logging(args.verbose)
    window_hours = _resolve_window_hours(args=args, parser=parser)

    try:
        status_document = await build_ops_status_document(
            window_hours=window_hours,
        )
        if args.json:
            print(_compact_json(status_document))
            return

        print(format_ops_status_document(status_document))
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
