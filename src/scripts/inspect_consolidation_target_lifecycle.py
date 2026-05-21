"""Inspect synthetic consolidation target lifecycle rows.

This command defaults to read-only diagnostics. Apply mode is an explicit
operator action for cleanup after consolidation target routing has stopped
creating synthetic user targets.

Typical use:
    python -m scripts.inspect_consolidation_target_lifecycle
    python -m scripts.inspect_consolidation_target_lifecycle --output path.json
    python -m scripts.inspect_consolidation_target_lifecycle --apply
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import UTC, datetime
from pathlib import Path

from scripts._db_export import (
    configure_logging,
    configure_stdout,
    default_output_path,
    load_project_env,
    write_json_export,
)
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import (
    apply_consolidation_target_lifecycle_cleanup,
    inspect_consolidation_target_lifecycle,
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        Configured parser for lifecycle diagnostics and approved cleanup.
    """

    parser = argparse.ArgumentParser(
        description="Inspect synthetic consolidation target lifecycle rows.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the approved synthetic-row cleanup after dry-run review.",
    )
    parser.add_argument("--output", type=Path, help="Destination JSON path.")
    parser.add_argument("--verbose", action="store_true", help="Show DB logs.")
    return parser


async def main() -> None:
    """Run lifecycle diagnostics or an approved cleanup action."""

    configure_stdout()
    parser = _build_parser()
    args = parser.parse_args()
    configure_logging(args.verbose)
    load_project_env()

    mode = "apply" if args.apply else "dry_run"
    output_path = args.output or default_output_path(
        "consolidation_target_lifecycle",
        mode,
    )
    try:
        if args.apply:
            storage_timestamp_utc = datetime.now(UTC).isoformat()
            report = await apply_consolidation_target_lifecycle_cleanup(
                storage_timestamp_utc=storage_timestamp_utc,
            )
        else:
            report = await inspect_consolidation_target_lifecycle()

        write_json_export(
            output_path=output_path,
            query={
                "script": "inspect_consolidation_target_lifecycle",
                "mode": mode,
            },
            records_key="report",
            records=report,
            exclude_fields=[],
        )
        print(f"wrote consolidation target lifecycle {mode} to {output_path}")
        if args.apply:
            print(f"apply_status: {report['apply_status']}")
            print(
                "synthetic_user_owned_rows_after: "
                f"{report['synthetic_user_owned_rows_after']}"
            )
            for name, count in report["before_counts"].items():
                print(f"before_{name}: {count}")
            for name, count in report["after_counts"].items():
                print(f"after_{name}: {count}")
        else:
            for name, count in report["counts"].items():
                print(f"{name}: {count}")
        if report.get("cleanup_blocked"):
            print("cleanup_blocked: true")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper for the async CLI."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
