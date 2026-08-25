"""Collect a read-only audit of the typed shared-memory scope contract."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from bson import json_util

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db.script_operations import export_memory_rows
from kazusa_ai_chatbot.time_boundary import storage_utc_now_iso

REPORT_VERSION = "character_memory_scope_audit.v1"
MANIFEST_VERSION = "character_memory_scope_manifest.v1"
DEFAULT_LIMIT = 2000
LEARNED_MEMORY_TYPES = frozenset({"fact", "defense_rule"})
LEARNED_AUTHORITY_SOURCE_PAIRS = frozenset({
    ("conversation_accepted", "conversation_extracted"),
    ("reflection_promoted", "reflection_inferred"),
})
PRIVACY_REVIEW_FIELDS = frozenset({
    "global_applicability",
    "target_specific_meaning_removed",
    "affects_identity_or_boundaries",
    "private_detail_risk",
    "user_details_removed",
    "boundary_assessment",
    "reviewer",
})

RowLoader = Callable[..., Awaitable[list[dict[str, Any]]]]


def _scrub_embedding(value: Any) -> Any:
    """Remove only derived vector data before hashing or exporting."""

    if isinstance(value, Mapping):
        return {
            str(key): _scrub_embedding(item)
            for key, item in value.items()
            if key != "embedding"
        }
    if isinstance(value, list):
        return [_scrub_embedding(item) for item in value]
    return value


def stable_memory_row_hash(row: Mapping[str, Any]) -> str:
    """Return a deterministic hash for one repository row."""

    rendered = json_util.dumps(
        _scrub_embedding(dict(row)),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def memory_content_hash(row: Mapping[str, Any]) -> str:
    """Return the exact content hash used by the apply manifest."""

    content = row.get("content")
    normalized = content if isinstance(content, str) else ""
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _text(value: object) -> str:
    """Return bounded scalar text from an untrusted row."""

    return value.strip() if isinstance(value, str) else ""


def _is_learned_candidate(row: Mapping[str, Any]) -> bool:
    """Return whether a row belongs to the W3 learned-memory audit lane."""

    pair = (
        _text(row.get("authority")),
        _text(row.get("source_kind")),
    )
    return (
        _text(row.get("status")) == "active"
        and pair in LEARNED_AUTHORITY_SOURCE_PAIRS
        and _text(row.get("memory_type")) in LEARNED_MEMORY_TYPES
    )


def _privacy_review_issues(row: Mapping[str, Any]) -> list[str]:
    """Return exact structural/privacy issues for one learned row."""

    issues: list[str] = []
    review = row.get("privacy_review")
    if not isinstance(review, Mapping):
        return ["privacy_review_missing"]
    review_keys = set(review)
    missing = PRIVACY_REVIEW_FIELDS - review_keys
    extra = review_keys - PRIVACY_REVIEW_FIELDS
    if missing:
        issues.append("privacy_review_incomplete")
    if extra:
        issues.append("privacy_review_extra_fields")
    if review.get("global_applicability") != "global":
        issues.append("global_applicability_not_global")
    if review.get("target_specific_meaning_removed") is not True:
        issues.append("target_specific_meaning_retained")
    if review.get("affects_identity_or_boundaries") is not False:
        issues.append("identity_or_boundary_effect")
    if review.get("private_detail_risk") != "low":
        issues.append("private_detail_risk_not_low")
    if review.get("user_details_removed") is not True:
        issues.append("user_details_retained")
    if not _text(review.get("boundary_assessment")):
        issues.append("boundary_assessment_missing")
    if review.get("reviewer") != "automated_llm":
        issues.append("reviewer_not_automated_llm")
    return sorted(set(issues))


def validate_learned_memory_row(row: Mapping[str, Any]) -> list[str]:
    """Return W3 certificate violations without interpreting row content."""

    issues: list[str] = []
    memory_unit_id = _text(row.get("memory_unit_id"))
    if not memory_unit_id:
        issues.append("memory_unit_id_missing")
    if _text(row.get("status")) != "active":
        issues.append("status_not_active")
    if _text(row.get("memory_type")) not in LEARNED_MEMORY_TYPES:
        issues.append("memory_type_not_learned")
    pair = (
        _text(row.get("authority")),
        _text(row.get("source_kind")),
    )
    if pair not in LEARNED_AUTHORITY_SOURCE_PAIRS:
        issues.append("authority_source_pair_not_learned")
    if "source_global_user_id" not in row:
        issues.append("source_global_user_id_missing")
    elif _text(row.get("source_global_user_id")):
        issues.append("source_global_user_id_not_empty")
    if not isinstance(row.get("content"), str) or not _text(row.get("content")):
        issues.append("content_missing")
    issues.extend(_privacy_review_issues(row))
    return sorted(set(issues))


def _row_audit(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a bounded readable audit entry for one learned row."""

    issues = validate_learned_memory_row(row)
    memory_type = _text(row.get("memory_type"))
    valid = not issues
    if valid and memory_type == "fact":
        classification = "valid_target_free_global_fact"
    elif valid and memory_type == "defense_rule":
        classification = "valid_target_free_conditional_self_guidance"
    else:
        classification = "unresolved_or_invalid_certificate"
    memory_unit_id = _text(row.get("memory_unit_id"))
    return {
        "memory_unit_id": memory_unit_id,
        "memory_name": _text(row.get("memory_name")),
        "memory_type": memory_type,
        "authority": _text(row.get("authority")),
        "source_kind": _text(row.get("source_kind")),
        "source_global_user_id": _text(row.get("source_global_user_id")),
        "status": _text(row.get("status")),
        "lineage_id": _text(row.get("lineage_id")),
        "version": row.get("version"),
        "updated_at": _text(row.get("updated_at")),
        "row_hash": stable_memory_row_hash(row),
        "content_hash": memory_content_hash(row),
        "privacy_review": dict(row.get("privacy_review", {}))
        if isinstance(row.get("privacy_review"), Mapping)
        else None,
        "evidence_refs": row.get("evidence_refs", []),
        "classification": classification,
        "issue_codes": issues,
        "source_document": _scrub_embedding(dict(row)),
    }


def build_scope_audit_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    generated_at: str | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Build a pure read-only report and exact lifecycle manifest."""

    candidates = [row for row in rows if _is_learned_candidate(row)]
    audited_rows = [_row_audit(row) for row in candidates]
    manifest: list[dict[str, Any]] = []
    for audited in audited_rows:
        if not audited["issue_codes"]:
            continue
        manifest.append({
            "memory_unit_id": audited["memory_unit_id"],
            "expected_row_hash": audited["row_hash"],
            "expected_content_hash": audited["content_hash"],
            "expected_status": "active",
            "authority": audited["authority"],
            "source_kind": audited["source_kind"],
            "memory_type": audited["memory_type"],
            "disposition": "reject",
            "reason_code": "w3_certificate_invalid",
            "issue_codes": list(audited["issue_codes"]),
            "expected_privacy_review": deepcopy(audited["privacy_review"]),
        })
    report = {
        "report_version": REPORT_VERSION,
        "manifest_version": MANIFEST_VERSION,
        "mode": "read_only_audit",
        "generated_at": generated_at or storage_utc_now_iso(),
        "source_collection": "memory",
        "row_limit": limit,
        "rows_exported": len(rows),
        "learned_candidates": len(candidates),
        "certified_rows": sum(not row["issue_codes"] for row in audited_rows),
        "manifest_rows": len(manifest),
        "writes_attempted": 0,
        "offline_semantic_review": {
            "status": "pending_external_review",
            "raw_output": None,
            "contract": "typed certificate and provenance are preserved for review",
        },
        "rows": audited_rows,
        "apply_manifest": manifest,
    }
    return report


async def collect_scope_audit_report(
    *,
    row_loader: RowLoader | None = None,
    limit: int = DEFAULT_LIMIT,
) -> dict[str, Any]:
    """Read shared memory through the public export boundary exactly once."""

    loader = row_loader or export_memory_rows
    rows = await loader(
        query_filter={},
        projection={"embedding": 0},
        limit=limit,
    )
    return build_scope_audit_report(rows, limit=limit)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write readable Extended JSON without mutating repository state."""

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
    """Print readable Extended JSON through UTF-8 console streams."""

    _configure_utf8_stdio()
    print(json_util.dumps(value, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the read-only audit CLI parser."""

    parser = argparse.ArgumentParser(
        description="Audit learned shared-memory scope without writes.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_artifacts")
        / "diagnostics"
        / "character_memory_scope_audit.json",
    )
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    return parser


async def async_main() -> None:
    """Run the read-only audit and close the public database client."""

    args = build_arg_parser().parse_args()
    try:
        report = await collect_scope_audit_report(limit=args.limit)
        _write_json(args.output, report)
        _print_json(report)
    finally:
        await close_db()


if __name__ == "__main__":
    asyncio.run(async_main())
