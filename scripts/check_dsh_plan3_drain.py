"""Read-only five-count drain gate for the Plan 3 legacy cutover."""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from kazusa_ai_chatbot.db._client import get_db
from kazusa_ai_chatbot.db.script_operations import count_dsh_plan3_drain_rows

_TERMINAL_LEDGER_STATUSES = {"completed", "rejected", "failed", "cancelled"}
_NONTERMINAL_LEDGER_STATUSES = {
    "created",
    "source_resolved",
    "evidence_collected",
    "proposal_ready",
    "awaiting_approval",
    "applying",
    "verifying",
    "repairing",
    "blocked",
}


async def collect_dsh_plan3_drain(
    *,
    legacy_coding_workspace_root: Path,
) -> dict[str, object]:
    """Collect the exact legacy counts without mutating Mongo or ledgers."""

    root = _contained_root(legacy_coding_workspace_root)
    database = await get_db()
    mongo_counts = await count_dsh_plan3_drain_rows(database)
    report = {
        "schema_version": "dsh_plan3_drain_report.v1",
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "counts": {
            **mongo_counts,
            "nonterminal_or_invalid_legacy_coding_runs": _count_legacy_ledgers(root),
        },
    }
    counts = report["counts"]
    assert isinstance(counts, Mapping)
    report["ready"] = all(int(value) == 0 for value in counts.values())
    return report


def _contained_root(value: Path) -> Path:
    """Require one explicit absolute workspace root."""

    root = Path(value)
    if not root.is_absolute():
        raise ValueError("legacy-coding-workspace-root must be absolute")
    return root.resolve()


def _count_legacy_ledgers(root: Path) -> int:
    """Count invalid or nonterminal ledgers below the exact product root."""

    runs_root = root / "coding_runs"
    if not runs_root.exists():
        return 0
    count = 0
    for run_directory in runs_root.iterdir():
        if not run_directory.is_dir():
            continue
        ledger = run_directory / "run.json"
        if not ledger.exists() or not _is_contained(ledger, root):
            continue
        try:
            payload = json.loads(ledger.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            count += 1
            continue
        if not isinstance(payload, Mapping):
            count += 1
            continue
        if payload.get("schema_version") != "coding_run.v1":
            count += 1
            continue
        status = payload.get("status")
        if not isinstance(status, str) or (
            status not in _TERMINAL_LEDGER_STATUSES
            and status not in _NONTERMINAL_LEDGER_STATUSES
        ):
            count += 1
            continue
        if status in _NONTERMINAL_LEDGER_STATUSES:
            count += 1
    return count


def _is_contained(path: Path, root: Path) -> bool:
    """Return whether a resolved ledger remains below the declared root."""

    try:
        path.resolve().relative_to(root)
    except ValueError:
        return False
    return True


async def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Run the read-only drain checker and print its bounded JSON report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--legacy-coding-workspace-root",
        required=True,
        type=Path,
    )
    parser.add_argument("--format", choices=("json",), default="json")
    args = parser.parse_args(argv)
    report = await collect_dsh_plan3_drain(
        legacy_coding_workspace_root=args.legacy_coding_workspace_root,
    )
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    asyncio.run(main())
