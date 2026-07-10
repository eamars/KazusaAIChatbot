"""Live MongoDB proof for accepted-task coding-run context lookup."""

from __future__ import annotations

from uuid import uuid4

import pytest

from kazusa_ai_chatbot.accepted_task import load_open_coding_run_contexts_for_scope
from kazusa_ai_chatbot.db import close_db
from kazusa_ai_chatbot.db import _client as db_client
from kazusa_ai_chatbot.db.accepted_tasks import ensure_accepted_task_indexes
from kazusa_ai_chatbot.accepted_task.models import ACCEPTED_TASKS_COLLECTION

pytestmark = pytest.mark.live_db

TEST_DATABASE_NAME = "_test_coding_agent_phase_c"
TEST_ROW_PREFIX = "phase-c-live-"


@pytest.fixture
async def live_accepted_task_collection():
    """Provide the dedicated accepted-task collection and restore DB bindings."""

    db_client._client = None
    db_client._db = None
    db_client._db_loop = None
    original_database_name = db_client.MONGODB_DB_NAME
    db_client.MONGODB_DB_NAME = TEST_DATABASE_NAME
    db = await db_client.get_db()
    collection = db[ACCEPTED_TASKS_COLLECTION]
    await collection.delete_many({
        "accepted_task_id": {"$regex": f"^{TEST_ROW_PREFIX}"},
    })
    try:
        yield collection
    finally:
        await collection.delete_many({
            "accepted_task_id": {"$regex": f"^{TEST_ROW_PREFIX}"},
        })
        await close_db()
        db_client.MONGODB_DB_NAME = original_database_name


async def test_live_coding_run_context_loader_is_indexed_scoped_and_bounded(
    live_accepted_task_collection,
) -> None:
    """The public loader collapses newest rows without leaking user scope."""

    await ensure_accepted_task_indexes()
    await ensure_accepted_task_indexes()
    index_rows = await live_accepted_task_collection.list_indexes().to_list(
        length=None,
    )
    index_names = {row["name"] for row in index_rows}
    assert "accepted_task_open_coding_run_context_lookup" in index_names

    run_rows = [
        _accepted_task_row("run-001", "2026-07-10T00:00:05Z"),
        _accepted_task_row("run-001", "2026-07-10T00:00:01Z"),
        _accepted_task_row("run-002", "2026-07-10T00:00:04Z"),
        _accepted_task_row("run-003", "2026-07-10T00:00:03Z"),
        _accepted_task_row("run-004", "2026-07-10T00:00:02Z"),
        _accepted_task_row(
            "foreign-channel",
            "2026-07-10T00:00:06Z",
            source_channel_id="debug:user:other",
        ),
        _accepted_task_row(
            "foreign-user",
            "2026-07-10T00:00:07Z",
            requester_global_user_id="other-user",
        ),
    ]
    await live_accepted_task_collection.insert_many(run_rows)

    contexts = await load_open_coding_run_contexts_for_scope(
        source_platform="debug",
        source_channel_id="debug:user:phase-c",
        requester_global_user_id="user-phase-c",
    )

    assert [context["coding_run_ref"] for context in contexts] == [
        "coding_run:run-001",
        "coding_run:run-002",
        "coding_run:run-003",
    ]


def _accepted_task_row(
    run_id: str,
    updated_at: str,
    *,
    source_channel_id: str = "debug:user:phase-c",
    requester_global_user_id: str = "user-phase-c",
) -> dict[str, object]:
    """Build one dedicated live-test accepted-task context row."""

    row = {
        "accepted_task_id": f"{TEST_ROW_PREFIX}{uuid4().hex}",
        "source_platform": "debug",
        "source_channel_id": source_channel_id,
        "requester_global_user_id": requester_global_user_id,
        "action_kind": "accepted_coding_task_request",
        "updated_at": updated_at,
        "coding_run_context": {
            "schema_version": "coding_run_context.v1",
            "coding_run_ref": f"coding_run:{run_id}",
            "status": "blocked",
            "objective_summary": f"Objective for {run_id}",
            "allowed_next_actions": ["respond_to_blocker", "status"],
            "active_blocker": {
                "blocker_kind": "environment",
                "question": "Install the dependency and reply.",
                "options": ["Installed"],
            },
            "followup_open": True,
            "updated_at": updated_at,
        },
    }
    return row
