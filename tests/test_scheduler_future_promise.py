"""Unit tests for the scheduler's ``future_promise`` handler and related
cancel-with-timestamp behaviour.

MongoDB is mocked via ``get_db`` — every test is offline.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import kazusa_ai_chatbot.scheduler as scheduler


# ── Helpers ────────────────────────────────────────────────────────


def _make_mock_db() -> MagicMock:
    scheduled_events = MagicMock()
    scheduled_events.insert_one = AsyncMock()
    scheduled_events.update_one = AsyncMock(return_value=MagicMock(modified_count=1))
    scheduled_events.delete_many = AsyncMock()

    memory = MagicMock()
    memory.update_one = AsyncMock()

    db = MagicMock()
    db.scheduled_events = scheduled_events
    db.memory = memory
    return db


@pytest.fixture(autouse=True)
def _reset_pending_tasks():
    scheduler._pending_tasks.clear()
    yield
    for task in list(scheduler._pending_tasks.values()):
        task.cancel()
    scheduler._pending_tasks.clear()


# ── Tests ──────────────────────────────────────────────────────────


class TestFuturePromiseHandlerRegistered:
    def test_handler_exists(self):
        assert "future_promise" in scheduler._handlers


class TestFuturePromiseHandler:
    async def test_marks_memory_fulfilled(self):
        db = _make_mock_db()
        event = {
            "event_id": "evt-1",
            "event_type": "future_promise",
            "target_global_user_id": "u1",
            "payload": {
                "promise_text": "I will help you tomorrow",
                "memory_id": "mem-abc",
                "original_input": "can you help me tomorrow?",
                "context_summary": "user asked for help",
            },
        }
        with patch("kazusa_ai_chatbot.scheduler.get_db", AsyncMock(return_value=db)):
            await scheduler._handle_future_promise(event)

        db.memory.update_one.assert_awaited_once_with(
            {"_id": "mem-abc"},
            {"$set": {"status": "fulfilled"}},
        )

    async def test_missing_memory_id_is_noop(self):
        db = _make_mock_db()
        event = {
            "event_id": "evt-2",
            "event_type": "future_promise",
            "target_global_user_id": "u1",
            "payload": {"promise_text": "nothing"},
        }
        with patch("kazusa_ai_chatbot.scheduler.get_db", AsyncMock(return_value=db)):
            await scheduler._handle_future_promise(event)

        db.memory.update_one.assert_not_awaited()

    async def test_missing_payload_is_noop(self):
        db = _make_mock_db()
        event = {
            "event_id": "evt-3",
            "event_type": "future_promise",
            "target_global_user_id": "u1",
        }
        with patch("kazusa_ai_chatbot.scheduler.get_db", AsyncMock(return_value=db)):
            await scheduler._handle_future_promise(event)

        db.memory.update_one.assert_not_awaited()


class TestCancelEventStampsCancelledAt:
    async def test_cancel_sets_status_and_timestamp(self):
        db = _make_mock_db()
        scheduler._pending_tasks["evt-x"] = asyncio.create_task(asyncio.sleep(999))

        with patch("kazusa_ai_chatbot.scheduler.get_db", AsyncMock(return_value=db)):
            ok = await scheduler.cancel_event("evt-x")

        assert ok is True

        # Inspect the mock call — second arg should contain both fields
        call_args = db.scheduled_events.update_one.call_args
        filter_arg, update_arg = call_args.args[0], call_args.args[1]
        assert filter_arg == {"event_id": "evt-x", "status": "pending"}
        set_block = update_arg["$set"]
        assert set_block["status"] == "cancelled"
        assert "cancelled_at" in set_block
        # ISO-8601 format sanity check
        assert "T" in set_block["cancelled_at"]


class TestScheduleEventPersistsWithDefaults:
    async def test_schedule_auto_fills_event_id_and_status(self):
        db = _make_mock_db()
        with patch("kazusa_ai_chatbot.scheduler.get_db", AsyncMock(return_value=db)):
            # Use a scheduled time far in the future so the task does not fire
            event = {
                "event_type": "future_promise",
                "target_global_user_id": "u1",
                "scheduled_at": "2099-01-01T00:00:00+00:00",
                "payload": {"promise_text": "x", "memory_id": "mem-x",
                            "original_input": "y", "context_summary": "z"},
            }
            eid = await scheduler.schedule_event(event)

        assert eid
        db.scheduled_events.insert_one.assert_awaited_once()
        persisted = db.scheduled_events.insert_one.await_args.args[0]
        assert persisted["event_id"] == eid
        assert persisted["status"] == "pending"
        assert "created_at" in persisted


class TestFireEventDispatchesToRegisteredHandler:
    async def test_unknown_event_type_is_skipped_not_error(self):
        db = _make_mock_db()
        event = {
            "event_id": "evt-unknown",
            "event_type": "totally_made_up",
            "target_global_user_id": "u1",
        }
        with patch("kazusa_ai_chatbot.scheduler.get_db", AsyncMock(return_value=db)):
            await scheduler._fire_event(event)

        # Should log warning and mark completed anyway
        assert db.scheduled_events.update_one.await_count == 2
        # last call flags completed
        last_call = db.scheduled_events.update_one.await_args_list[-1]
        assert last_call.args[1]["$set"]["status"] == "completed"

    async def test_future_promise_marks_completed_after_handler(self):
        db = _make_mock_db()
        event = {
            "event_id": "evt-fp",
            "event_type": "future_promise",
            "target_global_user_id": "u1",
            "payload": {"promise_text": "p", "memory_id": "mem-1",
                        "original_input": "i", "context_summary": "c"},
        }
        with patch("kazusa_ai_chatbot.scheduler.get_db", AsyncMock(return_value=db)):
            await scheduler._fire_event(event)

        # scheduled_events: "running" then "completed"
        updates = [c.args[1]["$set"]["status"] for c in db.scheduled_events.update_one.await_args_list]
        assert updates == ["running", "completed"]
        # memory: "fulfilled"
        db.memory.update_one.assert_awaited_once()
