"""Apply an exact reviewed shared-memory scope manifest safely."""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from bson import json_util

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import export_memory_rows
from kazusa_ai_chatbot.memory_evolution import reject_memory_unit
from kazusa_ai_chatbot.rag.cache2_runtime import get_rag_cache2_runtime
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso
from scripts.audit_character_memory_scope import (
    build_scope_audit_report,
    memory_content_hash,
    stable_memory_row_hash,
    validate_learned_memory_row,
)

MANIFEST_VERSION = "character_memory_scope_manifest.v1"

RowLoader = Callable[..., Awaitable[list[dict[str, Any]]]]
Rejecter = Callable[..., Awaitable[Mapping[str, Any]]]
Clock = Callable[[], str]


class ManifestDriftError(ValueError):
    """Raised before mutation when any approved row changed or disappeared."""


def _text(value: object) -> str:
    """Return stripped text from a manifest value."""

    return value.strip() if isinstance(value, str) else ""


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write a readable lifecycle report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json_util.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _configure_utf8_stdio() -> None:
    """Configure supported console streams for lossless JSON output."""

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8", errors="strict")


def _print_json(value: Mapping[str, Any]) -> None:
    """Print one lifecycle report through UTF-8 console streams."""

    _configure_utf8_stdio()
    print(json_util.dumps(value, ensure_ascii=False, indent=2))


def load_approved_manifest(path: Path) -> list[dict[str, Any]]:
    """Load exact reject entries from an audit report or manifest file."""

    parsed = json_util.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, Mapping):
        raise TypeError("approved memory manifest must be an object")
    if parsed.get("manifest_version") != MANIFEST_VERSION:
        raise ValueError("approved memory manifest version is invalid")
    raw_entries = parsed.get("apply_manifest")
    if not isinstance(raw_entries, list):
        raise TypeError("approved memory manifest must contain apply_manifest")
    raw_audit_rows = parsed.get("rows", [])
    if not isinstance(raw_audit_rows, list):
        raise TypeError("approved memory report rows must be a list")
    audited_reviews = {
        _text(row.get("memory_unit_id")): deepcopy(row.get("privacy_review"))
        for row in raw_audit_rows
        if isinstance(row, Mapping) and _text(row.get("memory_unit_id"))
    }
    entries: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, Mapping):
            raise TypeError("manifest entry must be an object")
        entry = dict(raw_entry)
        memory_unit_id = _text(entry.get("memory_unit_id"))
        if not memory_unit_id or memory_unit_id in seen_ids:
            raise ValueError("manifest memory_unit_id must be unique and non-empty")
        if entry.get("disposition") != "reject":
            raise ValueError("manifest disposition must be reject")
        if not _text(entry.get("expected_row_hash")):
            raise ValueError("manifest row hash is required")
        if not _text(entry.get("expected_content_hash")):
            raise ValueError("manifest content hash is required")
        if entry.get("expected_status") != "active":
            raise ValueError("manifest expected status must be active")
        for field_name in ("authority", "source_kind", "memory_type"):
            if not _text(entry.get(field_name)):
                raise ValueError(f"manifest {field_name} is required")
        issue_codes = entry.get("issue_codes")
        if not isinstance(issue_codes, list) or not issue_codes:
            raise ValueError("manifest issue_codes must be a non-empty list")
        if "expected_privacy_review" not in entry:
            if memory_unit_id not in audited_reviews:
                raise ValueError(
                    "manifest expected privacy review is required"
                )
            entry["expected_privacy_review"] = deepcopy(
                audited_reviews[memory_unit_id]
            )
        seen_ids.add(memory_unit_id)
        entries.append(entry)
    return entries


def _row_map(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index repository rows by stable memory-unit id."""

    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        memory_unit_id = _text(row.get("memory_unit_id"))
        if memory_unit_id:
            if memory_unit_id in indexed:
                raise ManifestDriftError(
                    f"duplicate repository memory_unit_id: {memory_unit_id}"
                )
            indexed[memory_unit_id] = dict(row)
    return indexed


def preflight_manifest_rows(
    entries: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate every approved row before any lifecycle mutation."""

    indexed = _row_map(rows)
    drift: list[str] = []
    for entry in entries:
        memory_unit_id = _text(entry.get("memory_unit_id"))
        row = indexed.get(memory_unit_id)
        if row is None:
            drift.append(f"{memory_unit_id}:missing")
            continue
        live_issues = validate_learned_memory_row(row)
        if not live_issues:
            drift.append(f"{memory_unit_id}:currently_certified")
        expected_review = entry.get("expected_privacy_review")
        if row.get("privacy_review") != expected_review:
            drift.append(f"{memory_unit_id}:privacy_review_changed")
        expected_issues = entry.get("issue_codes")
        if (
            not isinstance(expected_issues, list)
            or sorted(expected_issues) != live_issues
        ):
            drift.append(f"{memory_unit_id}:certificate_issues_changed")
        for field_name in ("authority", "source_kind", "memory_type"):
            if _text(row.get(field_name)) != _text(entry.get(field_name)):
                drift.append(f"{memory_unit_id}:{field_name}_changed")
        if _text(row.get("status")) != entry.get("expected_status"):
            drift.append(f"{memory_unit_id}:status_changed")
        expected_content_hash = _text(entry.get("expected_content_hash"))
        if memory_content_hash(row) != expected_content_hash:
            drift.append(f"{memory_unit_id}:content_hash_changed")
        expected_hash = _text(entry.get("expected_row_hash"))
        if stable_memory_row_hash(row) != expected_hash:
            drift.append(f"{memory_unit_id}:row_hash_changed")
    if drift:
        raise ManifestDriftError("approved manifest drift: " + ", ".join(drift))
    return {
        key: indexed[key]
        for key in indexed
        if any(
            _text(entry.get("memory_unit_id")) == key
            for entry in entries
        )
    }


def _cache_stats(
    cache_stats_provider: Callable[[], Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Read sanitized Cache2 statistics through its public runtime API."""

    if cache_stats_provider is not None:
        return dict(cache_stats_provider())
    return dict(get_rag_cache2_runtime().get_stats())


async def apply_approved_manifest(
    *,
    entries: Sequence[Mapping[str, Any]],
    row_loader: RowLoader | None = None,
    rejecter: Rejecter | None = None,
    clock: Clock | None = None,
    cache_stats_provider: Callable[[], Mapping[str, Any]] | None = None,
    backup_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Back up, drift-check, reject, and re-audit exact unchanged rows."""

    loader = row_loader or export_memory_rows
    lifecycle_rejecter = rejecter or reject_memory_unit
    timestamp = clock or storage_utc_now_iso
    rows = await loader(
        query_filter={},
        projection={"embedding": 0},
        limit=max(2000, len(entries) * 2),
    )
    try:
        indexed = preflight_manifest_rows(entries, rows)
    except ManifestDriftError as exc:
        blocked = {
            "report_version": MANIFEST_VERSION,
            "mode": "apply",
            "apply_status": "blocked_drift",
            "writes_attempted": 0,
            "blocked_reason": str(exc),
            "applied": [],
        }
        if output_path is not None:
            _write_json(output_path, blocked)
        raise

    backup = {
        "manifest_version": MANIFEST_VERSION,
        "mode": "backup",
        "rows": [deepcopy(indexed[_text(entry["memory_unit_id"])]) for entry in entries],
    }
    if backup_path is not None:
        _write_json(backup_path, backup)

    before_cache = _cache_stats(cache_stats_provider)
    applied: list[dict[str, Any]] = []
    for entry in entries:
        memory_unit_id = _text(entry["memory_unit_id"])
        rejected = await lifecycle_rejecter(
            active_unit_id=memory_unit_id,
            reason=_text(entry.get("reason_code")) or "w3_certificate_invalid",
            storage_timestamp_utc=timestamp(),
        )
        applied.append({
            "memory_unit_id": memory_unit_id,
            "status": rejected.get("status") if isinstance(rejected, Mapping) else "rejected",
            "reason_code": entry.get("reason_code", ""),
        })

    after_rows = await loader(
        query_filter={},
        projection={"embedding": 0},
        limit=max(2000, len(entries) * 2),
    )
    after_cache = _cache_stats(cache_stats_provider)
    post_audit = build_scope_audit_report(after_rows)
    report = {
        "report_version": MANIFEST_VERSION,
        "mode": "apply",
        "apply_status": "applied",
        "writes_attempted": len(applied),
        "applied": applied,
        "cache_verification": {
            "before": before_cache,
            "after": after_cache,
            "invalidations_increased": (
                after_cache.get("invalidations", 0)
                > before_cache.get("invalidations", 0)
            ),
        },
        "post_audit": post_audit,
    }
    if output_path is not None:
        _write_json(output_path, report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the exact-manifest apply CLI parser."""

    parser = argparse.ArgumentParser(
        description="Apply an approved unchanged shared-memory manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


async def async_main() -> None:
    """Apply one reviewed manifest and close the public DB client."""

    args = build_arg_parser().parse_args()
    entries = load_approved_manifest(args.manifest)
    try:
        report = await apply_approved_manifest(
            entries=entries,
            backup_path=args.backup,
            output_path=args.output,
        )
        _print_json(report)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(async_main())
