"""Guarded live-DB rehearsal for the task-history big-bang boundary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from uuid import uuid4

import pytest

from kazusa_ai_chatbot.accepted_task import lifecycle
from kazusa_ai_chatbot.config import MONGODB_DB_NAME
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db._client import TEST_DATABASE_NAME, get_db
from kazusa_ai_chatbot.db.accepted_tasks import ensure_accepted_task_indexes


pytestmark = [pytest.mark.asyncio, pytest.mark.live_db]

_ARTIFACT_ROOT = Path("test_artifacts/task_resolution/raw")
_TARGET_COLLECTIONS = ("background_work_jobs", "accepted_tasks")
_PRESERVED_COLLECTIONS = ("coding_runs", "calendar_schedules")


def _maintenance_module() -> ModuleType:
    """Load the reviewed maintenance boundary without running its CLI."""

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "clear_background_task_history.py"
    )
    spec = importlib.util.spec_from_file_location(
        "task_resolution_cutover_maintenance",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task_request(marker: str) -> dict[str, object]:
    """Build one target-state accepted-task request for the rehearsal."""

    return {
        "task_kind": "task_resolution",
        "semantic_objective": f"Verify v2 cutover task {marker}.",
        "accepted_task_summary": "Verify one v2 task after history cleanup.",
        "requested_delivery": "send_result_when_done",
        "max_output_chars": 3000,
        "source_trigger_source": "user_message",
        "source_platform": "debug",
        "source_channel_id": f"debug:user:{marker}",
        "source_channel_type": "private",
        "source_message_id": f"message-{marker}",
        "source_platform_bot_id": "debug-bot-task-resolution",
        "source_character_name": "Kazusa",
        "requester_global_user_id": f"global-user-{marker}",
        "requester_platform_user_id": f"debug-user-{marker}",
        "requester_display_name": "Test User",
        "storage_timestamp_utc": "2026-08-01T01:00:00+00:00",
    }


def _write_artifact(value: dict[str, object]) -> Path:
    """Write count and preservation evidence for review."""

    _ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    path = _ARTIFACT_ROOT / "cutover_live_db_rehearsal.json"
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


async def test_task_resolution_cutover_live_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Clear only task history, preserve markers, and complete one v2 task."""

    monkeypatch.setenv("KAZUSA_TEST_DB_GUARD", "1")
    assert MONGODB_DB_NAME == TEST_DATABASE_NAME
    marker = f"task-resolution-{uuid4().hex}"
    db = await get_db()
    maintenance = _maintenance_module()
    assert maintenance.TARGET_COLLECTIONS == _TARGET_COLLECTIONS
    try:
        await db["background_work_jobs"].insert_one({
            "_id": f"job-{marker}",
            "schema_version": "background_work_job.v1",
            "test_marker": marker,
        })
        await db["accepted_tasks"].insert_one({
            "_id": f"task-{marker}",
            "schema_version": "accepted_task.v1",
            "accepted_task_id": f"legacy-task-{marker}",
            "test_marker": marker,
        })
        for collection_name in _PRESERVED_COLLECTIONS:
            await db[collection_name].insert_one({
                "_id": marker,
                "schema_version": "preservation_marker.v1",
                "test_marker": marker,
            })

        dry_run = await maintenance.clear_background_task_history(
            execute=False
        )
        assert all(dry_run["before"][name] >= 1 for name in _TARGET_COLLECTIONS)
        execute = await maintenance.clear_background_task_history(execute=True)
        assert execute["remaining"] == {
            "background_work_jobs": 0,
            "accepted_tasks": 0,
        }
        preserved = {
            name: await db[name].count_documents({"test_marker": marker})
            for name in _PRESERVED_COLLECTIONS
        }
        assert preserved == {"coding_runs": 1, "calendar_schedules": 1}

        await ensure_accepted_task_indexes()
        created = await lifecycle.create_or_return_active_accepted_task(
            _task_request(marker)
        )
        accepted_task_id = created["task"]["accepted_task_id"]
        assert created["status"] == "created"
        pending = await lifecycle.mark_accepted_task_pending(
            accepted_task_id=accepted_task_id,
            executor_ref=f"job-v2-{marker}",
            updated_at="2026-08-01T01:00:01+00:00",
        )
        assert pending is not None
        running = await lifecycle.mark_accepted_task_running(
            accepted_task_id=accepted_task_id,
            started_at="2026-08-01T01:00:02+00:00",
        )
        assert running is not None
        completed = await lifecycle.mark_tool_result_ready(
            accepted_task_id=accepted_task_id,
            artifact_text="",
            result_summary="The v2 cutover rehearsal task completed.",
            completed_at="2026-08-01T01:00:03+00:00",
            result_kind="resolved",
            completion_status="resolved",
        )
        assert completed is not None
        assert completed["schema_version"] == "accepted_task.v2"
        assert completed["state"] == "result_ready"
        assert completed["completion_status"] == "resolved"
        assert await db["accepted_tasks"].count_documents({
            "schema_version": "accepted_task.v1"
        }) == 0

        artifact_path = _write_artifact({
            "schema_version": "task_resolution_cutover_rehearsal.v1",
            "database_name": MONGODB_DB_NAME,
            "marker": marker,
            "dry_run": dry_run,
            "execute": execute,
            "preserved_collection_counts": preserved,
            "first_v2_task": {
                "accepted_task_id": accepted_task_id,
                "state": completed["state"],
                "completion_status": completed["completion_status"],
            },
            "coding_agent_operation_called": False,
        })
        print(f"TASK_RESOLUTION_CUTOVER_ARTIFACT={artifact_path}")
    finally:
        await db["accepted_tasks"].delete_many({
            "requester_global_user_id": f"global-user-{marker}",
        })
        for collection_name in (
            *_TARGET_COLLECTIONS,
            *_PRESERVED_COLLECTIONS,
        ):
            await db[collection_name].delete_many({"test_marker": marker})
        await close_db()
