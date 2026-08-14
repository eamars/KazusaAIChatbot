"""Read-only cutover preflight for the scheduled future-speech contract.

    The preflight scans active future-speak accepted tasks, background jobs,
    and structurally identified future-speak calendar schedules and runs for the immutable
``scheduled_future_speech_authority`` schema and required source identity.
It performs no writes, migration, terminalization, or status changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping
from typing import Any

from kazusa_ai_chatbot.accepted_task.models import (
    ACCEPTED_TASKS_COLLECTION,
    ACTIVE_ACCEPTED_TASK_STATES,
    TERMINAL_ACCEPTED_TASK_STATES,
)
from kazusa_ai_chatbot.background_work.models import (
    BACKGROUND_WORK_JOBS_COLLECTION,
    FUTURE_SPEAK_WORKER,
)
from kazusa_ai_chatbot.calendar_scheduler import models as calendar_models
from kazusa_ai_chatbot.calendar_scheduler import repository as calendar_repository
from kazusa_ai_chatbot.cognition_core_v2.contracts import (
    CognitionContractError,
    validate_scheduled_future_speech_authority,
)
from kazusa_ai_chatbot.db._client import get_db

ACTIVE_JOB_STATES = ("queued", "in_progress", "delivery_in_progress")
TERMINAL_JOB_STATES = (
    "completed",
    "failed",
    "delivered",
    "delivery_failed",
)
ACTIVE_RUN_STATES = (calendar_models.RUN_STATUS_PENDING, calendar_models.RUN_STATUS_RUNNING)
TERMINAL_RUN_STATES = (
    calendar_models.RUN_STATUS_COMPLETED,
    calendar_models.RUN_STATUS_FAILED,
    calendar_models.RUN_STATUS_SKIPPED,
    calendar_models.RUN_STATUS_CANCELLED,
)
DEFAULT_SAMPLE_LIMIT = 5


def _authority_status(authority: object) -> tuple[bool, str]:
    """Return whether one authority payload is valid for the new writer."""

    if not isinstance(authority, Mapping):
        return False, "missing"
    try:
        validate_scheduled_future_speech_authority(authority)
    except (CognitionContractError, ValueError) as exc:
        return False, f"invalid: {exc}"
    return True, "valid"


async def _scan_collection(
    db: Any,
    collection_name: str,
    query: dict[str, Any],
    id_field: str,
    *,
    sample_limit: int,
    authority_in_payload: bool = False,
) -> dict[str, Any]:
    """Scan one collection and return bounded incompatible sample ids."""

    collection = db[collection_name]
    incompatible: list[str] = []
    incompatible_count = 0
    compatible_count = 0
    cursor = collection.find(query, {"_id": 0})
    async for row in cursor:
        if not isinstance(row, Mapping):
            continue
        if authority_in_payload:
            payload = row.get("payload")
            authority_value = (
                payload.get(calendar_models.SCHEDULED_AUTHORITY_PAYLOAD_KEY)
                if isinstance(payload, Mapping)
                else None
            )
        else:
            authority_value = row.get("scheduled_future_speech_authority")
        ok, _ = _authority_status(authority_value)
        row_id = str(row.get(id_field) or "")
        if ok:
            compatible_count += 1
            continue
        incompatible_count += 1
        if row_id and len(incompatible) < sample_limit:
            incompatible.append(row_id)
    result = {
        "compatible_active_count": compatible_count,
        "incompatible_active_count": incompatible_count,
        "incompatible_sample_ids": incompatible,
    }
    return result


async def _count_documents(
    db: Any,
    collection_name: str,
    query: dict[str, Any],
) -> int:
    """Count one read-only historical query."""

    collection = db[collection_name]
    count = await collection.count_documents(query)
    return int(count)


async def run_preflight(*, sample_limit: int = DEFAULT_SAMPLE_LIMIT) -> dict[str, Any]:
    """Run the read-only active legacy-record preflight.

    Args:
        sample_limit: Maximum incompatible ids retained per collection.

    Returns:
        A deterministic report with per-collection scans and a
        ``deployment_blocked`` decision. No database writes are performed.
    """

    db = await get_db()
    task_scan = await _scan_collection(
        db,
        ACCEPTED_TASKS_COLLECTION,
        {
            "task_kind": "future_speak",
            "state": {"$in": list(ACTIVE_ACCEPTED_TASK_STATES)},
        },
        "accepted_task_id",
        sample_limit=sample_limit,
    )
    job_scan = await _scan_collection(
        db,
        BACKGROUND_WORK_JOBS_COLLECTION,
        {
            "requested_worker": FUTURE_SPEAK_WORKER,
            "status": {"$in": list(ACTIVE_JOB_STATES)},
        },
        "job_id",
        sample_limit=sample_limit,
    )
    schedule_scan = await _scan_collection(
        db,
        calendar_repository.CALENDAR_SCHEDULES_COLLECTION,
        {
            "trigger_kind": calendar_models.TRIGGER_FUTURE_COGNITION,
            "status": calendar_models.SCHEDULE_STATUS_ACTIVE,
            "payload.source_refs.ref_id": (
                calendar_models.FUTURE_SPEAK_SOURCE_REF_ID
            ),
        },
        "schedule_id",
        sample_limit=sample_limit,
        authority_in_payload=True,
    )
    run_scan = await _scan_collection(
        db,
        calendar_repository.CALENDAR_RUNS_COLLECTION,
        {
            "trigger_kind": calendar_models.TRIGGER_FUTURE_COGNITION,
            "status": {"$in": list(ACTIVE_RUN_STATES)},
            "payload.source_refs.ref_id": (
                calendar_models.FUTURE_SPEAK_SOURCE_REF_ID
            ),
        },
        "run_id",
        sample_limit=sample_limit,
        authority_in_payload=True,
    )
    scans = {
        "accepted_tasks": task_scan,
        "background_work_jobs": job_scan,
        "calendar_schedules": schedule_scan,
        "calendar_runs": run_scan,
    }
    incompatible_active_count = sum(
        scan["incompatible_active_count"] for scan in scans.values()
    )
    historical_legacy_count = await _count_documents(
        db,
        ACCEPTED_TASKS_COLLECTION,
        {
            "task_kind": "future_speak",
            "state": {"$in": list(TERMINAL_ACCEPTED_TASK_STATES)},
        },
    )
    historical_legacy_count += await _count_documents(
        db,
        BACKGROUND_WORK_JOBS_COLLECTION,
        {
            "requested_worker": FUTURE_SPEAK_WORKER,
            "status": {"$in": list(TERMINAL_JOB_STATES)},
        },
    )
    historical_legacy_count += await _count_documents(
        db,
        calendar_repository.CALENDAR_SCHEDULES_COLLECTION,
        {
            "trigger_kind": calendar_models.TRIGGER_FUTURE_COGNITION,
            "status": {
                "$in": [
                    calendar_models.SCHEDULE_STATUS_COMPLETED,
                    calendar_models.SCHEDULE_STATUS_CANCELLED,
                ]
            },
            "payload.source_refs.ref_id": (
                calendar_models.FUTURE_SPEAK_SOURCE_REF_ID
            ),
        },
    )
    historical_legacy_count += await _count_documents(
        db,
        calendar_repository.CALENDAR_RUNS_COLLECTION,
        {
            "trigger_kind": calendar_models.TRIGGER_FUTURE_COGNITION,
            "status": {"$in": list(TERMINAL_RUN_STATES)},
            "payload.source_refs.ref_id": (
                calendar_models.FUTURE_SPEAK_SOURCE_REF_ID
            ),
        },
    )
    report = {
        "preflight": "scheduled_future_speech_contract",
        "mode": "read_only",
        "scans": scans,
        "incompatible_active_count": incompatible_active_count,
        "historical_legacy_count": historical_legacy_count,
        "deployment_blocked": incompatible_active_count > 0,
        "sample_limit": sample_limit,
    }
    return report


def exit_code_from_report(report: Mapping[str, Any]) -> int:
    """Return the process exit code for one preflight report."""

    if report.get("deployment_blocked") is True:
        exit_code = 1
    else:
        exit_code = 0
    return exit_code


def _configure_console_encoding() -> None:
    """Make CJK-safe stdout and stderr available on Windows."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


async def _async_main(sample_limit: int) -> int:
    """Run the preflight and print its bounded report."""

    report = await run_preflight(sample_limit=sample_limit)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["deployment_blocked"]:
        print(
            "DEPLOYMENT_BLOCKED: active legacy future-speak records lack the "
            "new scheduled authority schema.",
            file=sys.stderr,
        )
    return exit_code_from_report(report)


def main() -> int:
    """Entrypoint that returns a nonzero code when cutover is blocked."""

    _configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Read-only scheduled future-speech cutover preflight."
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=DEFAULT_SAMPLE_LIMIT,
        help="Maximum incompatible active ids retained per collection.",
    )
    args = parser.parse_args()
    exit_code = asyncio.run(_async_main(sample_limit=args.sample_limit))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
