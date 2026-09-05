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




