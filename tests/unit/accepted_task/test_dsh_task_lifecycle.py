"""Executable tests for accepted-task DSH affordances and lifecycle."""

from __future__ import annotations

import importlib
from typing import Any, get_args

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


def test_task_resolution_active_identity_uses_continuation_not_model_wording() -> None:
    """Task-resolution duplicate identity excludes paraphrasable objectives."""

    lifecycle = _module("kazusa_ai_chatbot.accepted_task.lifecycle")
    from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref

    continuation = build_goal_continuation_ref(
        source_episode_id="episode-task-001",
        source_message_id="message-1",
        branch_id="b1",
        goal_ref={
            "scope": "user",
            "kind": "goal",
            "entity_id": "goal-001",
        },
    )
    first = _task()
    first["goal_continuation_ref"] = continuation
    second = {
        **first,
        "semantic_objective": "A model-paraphrased form of the same task.",
        "accepted_task_summary": "A model-paraphrased form of the same task.",
    }

    assert lifecycle.build_task_identity_key(first) == (
        lifecycle.build_task_identity_key(second)
    )
    material = lifecycle._identity_material(first)
    assert material["task_kind"] == "task_resolution"
    assert material["goal_continuation_ref"] == continuation
    assert "semantic_objective" not in material


def test_future_speak_active_identity_retains_semantic_objective() -> None:
    """Future-speak scheduling continues to identify by semantic objective."""

    lifecycle = _module("kazusa_ai_chatbot.accepted_task.lifecycle")
    first = {
        "task_kind": "future_speak",
        "semantic_objective": "Remind the user at the agreed time.",
        "source_platform": "debug",
        "source_channel_id": "channel-1",
        "source_channel_type": "private",
        "requester_global_user_id": "user-1",
        "requester_platform_user_id": "debug-user-1",
        "goal_continuation_ref": None,
    }
    second = {
        **first,
        "semantic_objective": "Remind the user at another agreed time.",
    }

    first_material = lifecycle._identity_material(first)
    second_material = lifecycle._identity_material(second)
    assert first_material["task_kind"] == "future_speak"
    assert first_material["semantic_objective"] == (
        "Remind the user at the agreed time."
    )
    assert second_material["semantic_objective"] != (
        first_material["semantic_objective"]
    )
    assert lifecycle.build_task_identity_key(first) != (
        lifecycle.build_task_identity_key(second)
    )


def test_dsh_task_states_match_current_lifecycle() -> None:
    """The accepted-task state contract matches the current lifecycle."""

    models = _module("kazusa_ai_chatbot.accepted_task.models")
    assert set(get_args(models.AcceptedTaskState)) == {
        "enqueueing",
        "pending",
        "running",
        "result_ready",
        "failure_ready",
        "delivery_in_progress",
        "delivery_retryable",
        "delivered",
        "enqueue_failed",
        "delivery_exhausted",
        "cancelled",
        "superseded",
    }


def test_dsh_task_affordance_uses_opaque_ref_for_active_and_open_followup() -> None:
    """The real projection exposes only prompt-safe task affordance fields."""

    models = _module("kazusa_ai_chatbot.accepted_task.models")
    project = getattr(models, "project_dsh_task_affordance", None)
    if not callable(project):
        pytest.fail("accepted-task model owner lacks project_dsh_task_affordance")
    affordance = project(_task("delivered", followup_open=True), _binding())
    assert set(affordance) == {
        "schema_version",
        "accepted_task_ref",
        "task_state",
        "objective_summary",
        "latest_summary",
        "allowed_next_actions",
        "followup_open",
        "updated_at",
    }
    assert affordance["accepted_task_ref"] == "accepted_task:task-1"
    assert affordance["allowed_next_actions"] == ["continue", "summarize", "cancel"]
    serialized = repr(affordance)
    assert "session-1" not in serialized
    assert "thread-1" not in serialized
    assert "authority" not in serialized


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
