"""Decommission checks for the retired process-local delayed-task runtime."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src" / "kazusa_ai_chatbot"
LEGACY_COLLECTION_NAME = "scheduled" + "_events"
LEGACY_PENDING_INDEX_NAME = "Pending" + "TaskIndex"
LEGACY_ENABLE_FLAG = "SCHEDULED" + "_TASKS_ENABLED"
LEGACY_LOAD_FUNC = "load" + "_pending_events"
LEGACY_SCHEDULE_FUNC = "schedule" + "_event"
LEGACY_ROW_NAME = "scheduled" + "_event"


def test_legacy_process_local_runtime_modules_are_removed() -> None:
    """The old runtime modules should no longer exist after calendar cutover."""

    removed_paths = [
        SRC_ROOT / "scheduler.py",
        SRC_ROOT / "db" / "scheduled_events.py",
        SRC_ROOT / "dispatcher" / "pending_index.py",
    ]

    assert all(not path.exists() for path in removed_paths)


def test_service_lifespan_source_does_not_start_legacy_runtime() -> None:
    """Service startup should not configure or load the retired runtime."""

    service_source = (SRC_ROOT / "service.py").read_text(encoding="utf-8")
    forbidden_tokens = [
        LEGACY_ENABLE_FLAG,
        LEGACY_PENDING_INDEX_NAME,
        LEGACY_LOAD_FUNC,
        "configure_runtime",
        "scheduler.shutdown",
    ]

    assert all(token not in service_source for token in forbidden_tokens)


def test_db_public_facade_does_not_export_legacy_scheduler_helpers() -> None:
    """Legacy row helpers should not remain on the runtime DB facade."""

    from kazusa_ai_chatbot import db

    forbidden_exports = [
        "cancel_pending_" + LEGACY_ROW_NAME,
        "claim_pending_" + LEGACY_ROW_NAME + "_running",
        "insert_" + LEGACY_ROW_NAME,
        "list_due_future_cognition_events",
        "list_pending_scheduler_events",
        "mark_" + LEGACY_ROW_NAME + "_completed",
        "mark_" + LEGACY_ROW_NAME + "_failed",
        "mark_" + LEGACY_ROW_NAME + "_running",
        "query_pending_" + LEGACY_COLLECTION_NAME,
    ]

    assert all(not hasattr(db, export_name) for export_name in forbidden_exports)


def test_dispatcher_facade_does_not_export_legacy_pending_index() -> None:
    """Dispatcher remains an adapter/tool layer, not a delayed-task index."""

    from kazusa_ai_chatbot import dispatcher

    assert not hasattr(dispatcher, LEGACY_PENDING_INDEX_NAME)
