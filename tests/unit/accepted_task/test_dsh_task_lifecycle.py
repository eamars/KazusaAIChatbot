"""Executable tests for accepted-task DSH affordances and lifecycle."""

from __future__ import annotations

import importlib
from typing import Any

import pytest


def _task(state: str = "running", *, followup_open: bool = False) -> dict[str, object]:
    return {
        "schema_version": "accepted_task.v2",
        "accepted_task_id": "task-1",
        "task_kind": "task_resolution",
        "state": state,
        "semantic_objective": "Resolve one bounded goal.",
        "accepted_task_summary": "Resolve one bounded goal.",
        "result_summary": "The goal is complete.",
        "dsh_task_session_id": "session-1",
        "dsh_operation_generation": 0,
        "dsh_followup_open": followup_open,
        "dsh_followup_claim_action_attempt_id": None,
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "updated_at": "2026-08-30T22:00:00Z",
    }


def _binding() -> dict[str, object]:
    return {
        "schema_version": "dsh_task_binding.v1",
        "task_session_id": "session-1",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "operation_generation": 0,
        "state": "terminal",
        "revision": 3,
    }


def _module(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned accepted-task owner is unavailable: {module_name}: {exc}")










@pytest.mark.asyncio
async def test_dsh_promotion_and_followup_claim_preserve_one_session_with_new_delivery_row() -> None:
    """A follow-up creates one new delivery row for the same DSH session."""

    lifecycle = _module("kazusa_ai_chatbot.accepted_task.lifecycle")
    claim = getattr(lifecycle, "claim_dsh_followup", None)
    if not callable(claim):
        pytest.fail("accepted-task lifecycle lacks claim_dsh_followup")
    repository = _FakeRepository()
    first = await claim(
        control={
            "schema_version": "accepted_task_control.v1",
            "accepted_task_ref": "accepted_task:task-1",
            "operation": "continue",
            "instruction": "Continue with this semantic instruction.",
        },
        action_attempt_id="attempt-1",
        task=_task("delivered", followup_open=True),
        binding=_binding(),
        repository=repository,
    )
    retry = await claim(
        control={
            "schema_version": "accepted_task_control.v1",
            "accepted_task_ref": "accepted_task:task-1",
            "operation": "continue",
            "instruction": "Continue with this semantic instruction.",
        },
        action_attempt_id="attempt-1",
        task=_task("delivered", followup_open=True),
        binding=_binding(),
        repository=repository,
    )
    assert first["dsh_task_session_id"] == retry["dsh_task_session_id"] == "session-1"
    assert first["accepted_task_id"] == retry["accepted_task_id"]
    assert first["dsh_operation_generation"] == retry["dsh_operation_generation"] == 1
    assert repository.create_calls == 1


class _FakeRepository:
    def __init__(self) -> None:
        self.create_calls = 0
        self._followups: dict[tuple[str, str, int], dict[str, object]] = {}

    async def create_followup(self, **kwargs: object) -> dict[str, object]:
        key = (
            str(kwargs["task_session_id"]),
            str(kwargs["action_attempt_id"]),
            int(kwargs["operation_generation"]),
        )
        replay = self._followups.get(key)
        if replay is not None:
            return dict(replay)
        self.create_calls += 1
        result = {
            "accepted_task_id": "task-followup-1",
            "dsh_task_session_id": kwargs["task_session_id"],
            "dsh_operation_generation": 1,
            "state": "pending",
        }
        self._followups[key] = result
        return dict(result)
