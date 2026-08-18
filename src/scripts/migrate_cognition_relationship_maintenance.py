"""Backfill reducer-owned relationship maintenance metadata."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kazusa_ai_chatbot.cognition_core_v2.state_models import (
    RELATIONSHIP_MAINTENANCE_SCHEMA_VERSION,
    validate_cognition_state,
)
from kazusa_ai_chatbot.db.script_operations import (
    cognition_state_migration_digest,
    compare_and_replace_user_cognition_state_for_migration,
    list_user_cognition_states_for_relationship_maintenance_migration,
)

MIGRATION_SCHEMA_VERSION = "cognition_relationship_maintenance_migration.v1"
BACKUP_SCHEMA_VERSION = "cognition_relationship_maintenance_backup.v1"
MAX_MIGRATION_ERROR_CHARS = 500


def build_relationship_maintenance_state(
    state: dict[str, Any],
) -> dict[str, Any]:
    """Add an empty maintenance ledger without inferring historical events."""

    replacement = deepcopy(state)
    if replacement.get("state_scope") != "user":
        raise ValueError("relationship maintenance migration requires user state")
    relationship = replacement.get("relationship")
    if not isinstance(relationship, dict):
        raise TypeError(
            "relationship maintenance migration requires relationship"
        )
    relationship.setdefault(
        "relationship_maintenance",
        {
            "schema_version": RELATIONSHIP_MAINTENANCE_SCHEMA_VERSION,
            "last_interaction_date_utc": None,
            "last_bonus_date_utc": None,
            "last_source_id": None,
            "processed_source_ids": [],
        },
    )
    validated_replacement = validate_cognition_state(replacement)
    return validated_replacement


def build_dry_run_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build deterministic reviewed migration rows in user-id order."""

    reviewed_rows: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: str(item["global_user_id"])):
        reviewed_row: dict[str, Any] = {
            "global_user_id": row.get("global_user_id"),
            "status": "invalid",
        }
        current_state = row.get("cognition_state")
        if not isinstance(current_state, dict):
            reviewed_row["error"] = (
                "migration row cognition state must be an object"
            )
            reviewed_rows.append(reviewed_row)
            continue
        reviewed_row.update({
            "expected_previous_state": current_state,
            "expected_previous_digest": cognition_state_migration_digest(
                current_state
            ),
        })
        try:
            replacement = build_relationship_maintenance_state(current_state)
        except (KeyError, TypeError, ValueError) as exc:
            reviewed_row["error"] = str(exc)[:MAX_MIGRATION_ERROR_CHARS]
            reviewed_rows.append(reviewed_row)
            continue
        has_maintenance = "relationship_maintenance" in current_state[
            "relationship"
        ]
        reviewed_row.update({
            "status": "already_valid" if has_maintenance else "ready",
            "replacement_state": replacement,
        })
        reviewed_rows.append(reviewed_row)
    return reviewed_rows


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    """Write canonical JSON and return the exact bytes written."""

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return encoded


async def run_dry_run(
    *,
    backup_path: Path,
    report_path: Path,
    generated_at: str,
) -> dict[str, Any]:
    """Export a zero-write backup and migration report."""

    rows = await list_user_cognition_states_for_relationship_maintenance_migration()
    reviewed_rows = build_dry_run_rows(rows)
    backup = {
        "schema_version": BACKUP_SCHEMA_VERSION,
        "generated_at": generated_at,
        "rows": reviewed_rows,
    }
    backup_bytes = _write_json(backup_path, backup)
    report = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "mode": "dry_run",
        "generated_at": generated_at,
        "backup_path": str(backup_path),
        "backup_sha256": hashlib.sha256(backup_bytes).hexdigest(),
        "row_count": len(reviewed_rows),
        "writes_performed": 0,
        "rows": [
            {
                "global_user_id": row["global_user_id"],
                "status": row["status"],
                **(
                    {
                        "expected_previous_digest": row[
                            "expected_previous_digest"
                        ]
                    }
                    if "expected_previous_digest" in row
                    else {}
                ),
                **(
                    {
                        "replacement_digest": cognition_state_migration_digest(
                            row["replacement_state"]
                        )
                    }
                    if "replacement_state" in row
                    else {"error": row["error"]}
                ),
            }
            for row in reviewed_rows
        ],
    }
    _write_json(report_path, report)
    return report


async def run_apply(
    *,
    backup_path: Path,
    report_path: Path,
    output_path: Path,
    applied_at: str,
) -> dict[str, Any]:
    """Apply only rows whose complete reviewed state and digest still match."""

    backup_bytes = backup_path.read_bytes()
    backup = json.loads(backup_bytes.decode("utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if backup.get("schema_version") != BACKUP_SCHEMA_VERSION:
        raise ValueError("migration backup schema is invalid")
    if report.get("schema_version") != MIGRATION_SCHEMA_VERSION:
        raise ValueError("migration report schema is invalid")
    if report.get("mode") != "dry_run":
        raise ValueError("migration report is not a dry-run report")
    expected_backup_digest = report.get("backup_sha256")
    actual_backup_digest = hashlib.sha256(backup_bytes).hexdigest()
    if expected_backup_digest != actual_backup_digest:
        raise ValueError("migration backup digest does not match report")
    rows = backup.get("rows")
    if not isinstance(rows, list):
        raise TypeError("migration backup rows must be a list")

    current_rows = (
        await list_user_cognition_states_for_relationship_maintenance_migration()
    )
    current_by_user = {
        row["global_user_id"]: row
        for row in current_rows
        if isinstance(row.get("global_user_id"), str)
    }
    counts = {
        "updated": 0,
        "already_valid": 0,
        "drift": 0,
        "missing": 0,
        "failed": 0,
    }
    applied_rows: list[dict[str, str]] = []
    failed_rows: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            counts["failed"] += 1
            failed_rows.append({
                "disposition": "invalid",
                "error": "migration backup row is invalid",
            })
            continue
        status = row.get("status")
        user_id = row.get("global_user_id")
        if (
            not isinstance(user_id, str)
            or status not in {"already_valid", "ready"}
        ):
            counts["failed"] += 1
            failed_rows.append({
                "global_user_id": str(user_id or ""),
                "disposition": "invalid",
                "error": str(row.get("error", "migration row is invalid"))[
                    :MAX_MIGRATION_ERROR_CHARS
                ],
            })
            continue
        current = current_by_user.get(user_id)
        if not isinstance(current, dict):
            counts["missing"] += 1
            failed_rows.append({
                "global_user_id": user_id,
                "disposition": "missing",
                "error": "reviewed cognition state is missing",
            })
            continue
        current_state = current.get("cognition_state")
        expected_state = row.get("expected_previous_state")
        replacement = row.get("replacement_state")
        expected_digest = row.get("expected_previous_digest")
        if (
            not isinstance(current_state, dict)
            or not isinstance(expected_state, dict)
            or not isinstance(replacement, dict)
            or not isinstance(expected_digest, str)
            or current_state != expected_state
            or cognition_state_migration_digest(expected_state)
            != expected_digest
        ):
            counts["drift"] += 1
            failed_rows.append({
                "global_user_id": user_id,
                "disposition": "drift",
                "error": "current cognition state differs from dry-run state",
            })
            continue
        try:
            validate_cognition_state(replacement)
            if status == "already_valid":
                validate_cognition_state(current_state)
        except (KeyError, TypeError, ValueError) as exc:
            counts["failed"] += 1
            failed_rows.append({
                "global_user_id": user_id,
                "disposition": "invalid",
                "error": str(exc)[:MAX_MIGRATION_ERROR_CHARS],
            })
            continue
        if status == "already_valid":
            counts["already_valid"] += 1
            applied_rows.append({
                "global_user_id": user_id,
                "status": "already_valid",
                "replacement_digest": cognition_state_migration_digest(
                    replacement
                ),
            })
            continue
        try:
            committed = await (
                compare_and_replace_user_cognition_state_for_migration(
                    global_user_id=user_id,
                    # The reviewed values and digest were matched above. Use
                    # the freshly read document for Mongo's exact embedded
                    # document selector so BSON field ordering is preserved.
                    expected_previous_state=current_state,
                    expected_previous_digest=expected_digest,
                    replacement_state=replacement,
                )
            )
        except ValueError as exc:
            counts["failed"] += 1
            failed_rows.append({
                "global_user_id": user_id,
                "disposition": "invalid",
                "error": str(exc)[:MAX_MIGRATION_ERROR_CHARS],
            })
            continue
        if not committed:
            counts["drift"] += 1
            failed_rows.append({
                "global_user_id": user_id,
                "disposition": "drift",
                "error": "compare-and-set rejected the reviewed state",
            })
            continue
        counts["updated"] += 1
        applied_rows.append({
            "global_user_id": user_id,
            "status": "updated",
            "replacement_digest": cognition_state_migration_digest(
                replacement
            ),
        })
    result = {
        "schema_version": MIGRATION_SCHEMA_VERSION,
        "mode": "apply",
        "applied_at": applied_at,
        "backup_path": str(backup_path),
        "report_path": str(report_path),
        "backup_sha256": actual_backup_digest,
        "counts": counts,
        "activation_ready": not any(
            counts[key] for key in ("drift", "missing", "failed")
        ),
        "writes_performed": counts["updated"],
        "applied_rows": applied_rows,
        "failed_rows": failed_rows,
    }
    _write_json(output_path, result)
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the explicit dry-run/apply command contract."""

    parser = argparse.ArgumentParser(
        description="Backfill cognition relationship maintenance metadata.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    """Run the selected migration mode with one canonical timestamp."""

    generated_at = datetime.now(timezone.utc).isoformat(
        timespec="seconds"
    ).replace("+00:00", "Z")
    if args.dry_run:
        return await run_dry_run(
            backup_path=args.backup,
            report_path=args.report,
            generated_at=generated_at,
        )
    return await run_apply(
        backup_path=args.backup,
        report_path=args.report,
        output_path=args.output,
        applied_at=generated_at,
    )


def main() -> None:
    """Parse arguments and execute the migration command."""

    args = build_arg_parser().parse_args()
    result = asyncio.run(_main_async(args))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if result.get("mode") == "apply" and not result.get(
        "activation_ready",
        False,
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
