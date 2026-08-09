"""Export a bounded protected trace-correlation manifest."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from pymongo.errors import PyMongoError

from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db import script_operations
from kazusa_ai_chatbot.llm_tracing.correlation import (
    MAX_COMPANION_CANDIDATES,
    MAX_FAILURE_CAPSULES,
    MAX_PARENT_CANDIDATES,
    SOURCE_SURFACES,
    TraceCorrelationResolution,
    build_trace_correlation_manifest as build_manifest,
    normalize_identifier,
    normalize_source_surface,
    resolve_trace_candidates,
    safe_identifier_row,
    unique_trace_ids,
)
from kazusa_ai_chatbot.time_boundary import storage_utc_now, storage_utc_now_iso


def _configure_stdout() -> None:
    """Prefer UTF-8 terminal output when available."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _default_output_path(identifier: str) -> Path:
    """Build a safe default manifest path."""

    timestamp_utc = storage_utc_now().strftime("%Y%m%dT%H%M%SZ")
    safe_identifier = "".join(
        char if char.isalnum() or char in ("-", "_") else "_"
        for char in identifier
    )
    return Path("test_artifacts") / "diagnostics" / (
        f"trace_correlation_{safe_identifier}_{timestamp_utc}.json"
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the bounded manifest CLI parser."""

    parser = argparse.ArgumentParser(
        description="Export a bounded protected trace-correlation manifest."
    )
    parser.add_argument("--identifier", required=True)
    parser.add_argument(
        "--source-surface",
        choices=SOURCE_SURFACES,
        required=True,
    )
    parser.add_argument("--trace-id", default="")
    parser.add_argument("--cognition-invocation-id", default="")
    parser.add_argument("--output", type=Path)
    return parser


def _values_from_rows(
    rows: list[dict[str, Any]],
    *field_names: str,
) -> tuple[str, ...]:
    """Collect stable string fields from bounded rows."""

    values = [row.get(field_name, "") for row in rows for field_name in field_names]
    return unique_trace_ids(values)


def _identifier_entry(
    *,
    name: str,
    values: tuple[str, ...],
    owner: str,
    collection: str,
    field: str,
    absent_status: str = "not_found",
) -> dict[str, Any]:
    """Build one identifier status entry without selecting a candidate."""

    if len(values) == 1:
        status = "confirmed"
        value = values[0]
    elif len(values) > 1:
        status = "ambiguous"
        value = ""
    else:
        status = absent_status
        value = ""
    return {
        "name": name,
        "value": value,
        "candidate_values": list(values[:MAX_COMPANION_CANDIDATES]),
        "status": status,
        "owner": owner,
        "source": {"collection": collection, "field": field},
    }


def _join_entry(
    *,
    name: str,
    rows: list[dict[str, Any]],
    collection: str,
    relation: str,
    limit: int,
) -> dict[str, Any]:
    """Build one bounded exact-join record."""

    bounded = rows[:limit]
    conflict_count = sum(
        1
        for row in bounded
        if row.get("correlation_write_status") == "conflict"
    )
    if conflict_count:
        status = "conflict"
    elif bounded:
        status = "confirmed"
    else:
        status = "not_found"
    return {
        "status": status,
        "match_count": len(bounded),
        "capped": len(rows) >= limit,
        "conflict_count": conflict_count,
        "collection": collection,
        "relation": relation,
        "candidates": [
            safe_identifier_row(
                row,
                owner=collection,
                collection=collection,
            )
            for row in bounded
        ],
    }


_SOURCE_IDENTIFIER_FIELDS = {
    "protected_action_attempt_id": (
        "action_attempt_id",
        "self_cognition_action_attempts",
        "attempt_id",
    ),
    "protected_background_work_job_id": (
        "background_work_job_id",
        "background_work_jobs",
        "job_id",
    ),
    "protected_accepted_task_id": (
        "accepted_task_id",
        "background_work_jobs",
        "accepted_task_id",
    ),
    "protected_calendar_schedule_id": (
        "calendar_schedule_id",
        "calendar_schedules",
        "schedule_id",
    ),
    "protected_calendar_run_id": (
        "calendar_run_id",
        "calendar_runs",
        "run_id",
    ),
}


async def _resolve_parent(
    *,
    source_surface: str,
    identifier: str,
    explicit_trace_id: str,
) -> tuple[
    TraceCorrelationResolution,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Resolve the parent trace through the maintenance read boundary."""

    normalized_surface = normalize_source_surface(source_surface)
    normalized_identifier = normalize_identifier(identifier)
    if explicit_trace_id.strip():
        explicit = normalize_identifier(explicit_trace_id)
        rows = await script_operations.export_collection_rows(
            collection_name="llm_trace_runs",
            filter_doc={"trace_id": explicit},
            projection={
                "_id": 0,
                "trace_id": 1,
                "status": 1,
                "platform": 1,
                "platform_channel_id": 1,
                "channel_type": 1,
                "platform_message_id": 1,
                "global_user_id": 1,
                "delivery_tracking_id": 1,
                "started_at": 1,
                "completed_at": 1,
            },
            sort_doc={"started_at": 1},
            limit=MAX_PARENT_CANDIDATES,
        )
        resolution = resolve_trace_candidates(
            source_surface="protected_llm_trace_id",
            identifier=explicit,
            rows=rows,
        )
        resolution = TraceCorrelationResolution(
            source_surface=normalized_surface,
            identifier=normalized_identifier,
            status=resolution.status,
            trace_ids=resolution.trace_ids,
            reason="explicit trace-id override",
        )
        return resolution, rows if resolution.trace_id else [], []

    if normalized_surface == "unknown":
        return (
            resolve_trace_candidates(
                source_surface=normalized_surface,
                identifier=normalized_identifier,
                rows=[],
            ),
            [],
            [],
        )

    rows = await script_operations.list_trace_correlation_candidates(
        source_surface=normalized_surface,
        identifier=normalized_identifier,
    )
    resolution = resolve_trace_candidates(
        source_surface=normalized_surface,
        identifier=normalized_identifier,
        rows=rows,
    )
    if resolution.trace_id:
        parent_rows = await script_operations.export_collection_rows(
            collection_name="llm_trace_runs",
            filter_doc={"trace_id": resolution.trace_id},
            projection={
                "_id": 0,
                "trace_id": 1,
                "status": 1,
                "platform": 1,
                "platform_channel_id": 1,
                "channel_type": 1,
                "platform_message_id": 1,
                "global_user_id": 1,
                "delivery_tracking_id": 1,
                "started_at": 1,
                "completed_at": 1,
            },
            sort_doc={"started_at": 1},
            limit=MAX_PARENT_CANDIDATES,
        )
        return resolution, parent_rows, rows
    return resolution, [], rows


def _build_identifier_sections(
    *,
    parent_rows: list[dict[str, Any]],
    companions: dict[str, list[dict[str, Any]]],
    resolution: TraceCorrelationResolution,
    source_rows: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build canonical identifier and exact-join sections."""

    identifiers: dict[str, dict[str, Any]] = {}
    joins: dict[str, dict[str, Any]] = {}
    parent = parent_rows[0] if parent_rows else {}
    parent_available = resolution.status == "confirmed" and bool(parent_rows)
    derived_absent_status = (
        "not_captured" if parent_available else "not_applicable"
    )
    global_user_id = str(parent.get("global_user_id", "")).strip()
    identifiers["llm_trace_id"] = _identifier_entry(
        name="llm_trace_id",
        values=_values_from_rows(parent_rows, "trace_id"),
        owner="llm_trace_run",
        collection="llm_trace_runs",
        field="trace_id",
    )
    identifiers["global_user_id"] = _identifier_entry(
        name="global_user_id",
        values=(global_user_id,) if global_user_id else (),
        owner="llm_trace_run",
        collection="llm_trace_runs",
        field="global_user_id",
        absent_status=derived_absent_status,
    )

    step_rows = companions.get("llm_trace_steps", [])
    invocation_values = unique_trace_ids(
        [
            row.get("cognition_invocation_id", "")
            or (
                row.get("capsule", {})
                if isinstance(row.get("capsule"), dict)
                else {}
            ).get("cognition_invocation_id", "")
            for row in step_rows
        ]
    )
    identifiers["cognition_invocation_id"] = _identifier_entry(
        name="cognition_invocation_id",
        values=invocation_values,
        owner="llm_trace_step",
        collection="llm_trace_steps",
        field="capsule.cognition_invocation_id",
        absent_status=derived_absent_status,
    )

    companion_fields = (
        (
            "action_attempt_id",
            "self_cognition_action_attempts",
            "attempt_id",
        ),
        ("background_work_job_id", "background_work_jobs", "job_id"),
        ("accepted_task_id", "background_work_jobs", "accepted_task_id"),
        ("calendar_schedule_id", "calendar_schedules", "schedule_id"),
        ("calendar_run_id", "calendar_runs", "run_id"),
    )
    for name, collection, field in companion_fields:
        rows = companions.get(collection, [])
        identifiers[name] = _identifier_entry(
            name=name,
            values=_values_from_rows(rows, field),
            owner=collection,
            collection=collection,
            field=field,
            absent_status=derived_absent_status,
        )

    source_spec = _SOURCE_IDENTIFIER_FIELDS.get(resolution.source_surface)
    if source_spec is not None:
        name, collection, field = source_spec
        identifiers[name] = _identifier_entry(
            name=name,
            values=_values_from_rows(source_rows, field),
            owner=collection,
            collection=collection,
            field=field,
        )
    if resolution.status != "confirmed":
        identifiers["llm_trace_id"]["status"] = resolution.status
        identifiers["llm_trace_id"]["candidate_values"] = list(
            resolution.trace_ids[:MAX_PARENT_CANDIDATES]
        )

    joins["conversation_history"] = _join_entry(
        name="conversation_history",
        rows=companions.get("conversation_history", []),
        collection="conversation_history",
        relation="conversation_history.llm_trace_id == parent_trace.trace_id",
        limit=64,
    )
    joins["cognition_failure_capsules"] = _join_entry(
        name="cognition_failure_capsules",
        rows=[
            row
            for row in step_rows
            if row.get("capture_reason") == "cognition_failure_capsule"
        ],
        collection="llm_trace_steps",
        relation="llm_trace_steps.trace_id == parent_trace.trace_id",
        limit=MAX_FAILURE_CAPSULES,
    )
    for collection in (
        "self_cognition_action_attempts",
        "background_work_jobs",
        "calendar_schedules",
        "calendar_runs",
        "child_trace_runs",
    ):
        joins[collection] = _join_entry(
            name=collection,
            rows=companions.get(collection, []),
            collection=collection,
            relation=(
                "source_llm_trace_id == parent_trace.trace_id"
                if collection != "child_trace_runs"
                else "parent_llm_trace_id == parent_trace.trace_id"
            ),
            limit=MAX_COMPANION_CANDIDATES,
        )
    if not parent_available:
        for join in joins.values():
            join["status"] = "not_applicable"
            join["match_count"] = 0
            join["capped"] = False
            join["candidates"] = []
    return identifiers, joins


async def build_correlation_manifest(
    *,
    identifier: str,
    source_surface: str,
    explicit_trace_id: str = "",
    cognition_invocation_id: str = "",
) -> dict[str, Any]:
    """Build one bounded manifest, including explicit unavailable outcomes."""

    normalized_surface = normalize_source_surface(source_surface)
    normalized_identifier = normalize_identifier(identifier)
    try:
        resolution, parent_rows, source_rows = await _resolve_parent(
            source_surface=normalized_surface,
            identifier=normalized_identifier,
            explicit_trace_id=explicit_trace_id,
        )
    except (PyMongoError, OSError, TimeoutError) as exc:
        resolution = TraceCorrelationResolution(
            source_surface=normalized_surface,
            identifier=normalized_identifier,
            status="not_available",
            reason=f"protected read unavailable: {exc.__class__.__name__}",
        )
        parent_rows = []
        source_rows = []

    companions: dict[str, list[dict[str, Any]]] = {}
    unresolved: list[dict[str, Any]] = []
    if resolution.status == "conflict":
        unresolved.append({
            "relation": "parent_trace",
            "status": "conflict",
            "reason": resolution.reason,
        })
    if resolution.trace_id:
        try:
            companions = (
                await script_operations.list_trace_correlation_companions(
                    trace_id=resolution.trace_id,
                )
            )
        except (PyMongoError, OSError, TimeoutError) as exc:
            unresolved.append({
                "relation": "companion_reads",
                "status": "not_available",
                "reason": (
                    "protected read unavailable: "
                    f"{exc.__class__.__name__}"
                ),
            })

    identifiers, joins = _build_identifier_sections(
        parent_rows=parent_rows,
        companions=companions,
        resolution=resolution,
        source_rows=source_rows,
    )
    for relation, join in joins.items():
        if join.get("status") == "conflict":
            unresolved.append({
                "relation": relation,
                "status": "conflict",
                "reason": (
                    "durable source conflict is retained without selecting "
                    "a replacement trace"
                ),
            })
    if cognition_invocation_id.strip():
        selected = cognition_invocation_id.strip()
        selected_entry = identifiers["cognition_invocation_id"]
        if selected not in selected_entry["candidate_values"]:
            unresolved.append({
                "relation": "cognition_invocation_id",
                "identifier": selected,
                "status": "not_found",
                "reason": (
                    "selected invocation is not present in retained capsules"
                ),
            })
        else:
            selected_entry["selected"] = selected

    manifest = build_manifest(
        generated_at=storage_utc_now_iso(),
        resolution=resolution,
        explicit_trace_id=explicit_trace_id.strip(),
        cognition_invocation_id=cognition_invocation_id.strip(),
        parent_rows=parent_rows,
        identifiers=identifiers,
        joins=joins,
        unresolved=unresolved,
    )
    return manifest


def write_manifest(*, output_path: Path, manifest: dict[str, Any]) -> None:
    """Write one manifest JSON artifact."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


async def main() -> None:
    """Run the manifest exporter CLI."""

    _configure_stdout()
    args = _build_parser().parse_args()
    try:
        manifest = await build_correlation_manifest(
            identifier=args.identifier,
            source_surface=args.source_surface,
            explicit_trace_id=args.trace_id,
            cognition_invocation_id=args.cognition_invocation_id,
        )
        output_path = args.output or _default_output_path(args.identifier)
        write_manifest(output_path=output_path, manifest=manifest)
        print(f"wrote trace correlation manifest to {output_path}")
    finally:
        await close_db()


def async_main() -> None:
    """Console-script wrapper."""

    asyncio.run(main())


if __name__ == "__main__":
    async_main()
