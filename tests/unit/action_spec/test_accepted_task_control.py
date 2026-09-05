"""Executable tests for typed accepted-task controls."""

from __future__ import annotations

import importlib
from typing import Any
from unittest.mock import AsyncMock

import pytest


def _control(operation: str = "continue") -> dict[str, object]:
    return {
        "schema_version": "accepted_task_control.v1",
        "accepted_task_ref": "accepted_task:task-1",
        "operation": operation,
        "instruction": (
            "Please continue with this semantic instruction."
            if operation == "continue" else None
        ),
    }


def _affordance() -> dict[str, object]:
    return {
        "schema_version": "dsh_accepted_task_affordance.v1",
        "accepted_task_ref": "accepted_task:task-1",
        "task_state": "delivered",
        "objective_summary": "Resolve one bounded goal.",
        "latest_summary": "The goal is complete.",
        "allowed_next_actions": ["continue", "summarize", "cancel"],
        "followup_open": True,
        "updated_at": "2026-08-30T22:00:00Z",
    }


def _action_spec() -> dict[str, object]:
    return {
        "schema_version": "action_spec.v1",
        "kind": "accepted_task_control",
        "cognition_mode": "deliberative",
        "source_refs": [{
            "schema_version": "action_source_ref.v1",
            "ref_kind": "cognitive_episode",
            "ref_id": "episode-1",
            "owner": "cognition",
            "relationship": "basis",
            "evidence_refs": [],
        }],
        "target": {
            "schema_version": "action_target.v1",
            "target_kind": "none",
            "target_id": None,
            "owner": "accepted_task",
            "scope": {"platform": "debug", "channel_id": "channel-1"},
        },
        "params": {"control": _control()},
        "urgency": "background",
        "visibility": "private",
        "deadline": None,
        "continuation": {
            "schema_version": "action_continuation.v1",
            "mode": "background_followup",
            "episode_type": "task_resolution",
            "max_depth": 1,
            "include_result_as": "tool_result",
        },
        "surface_role": "task_acknowledgement",
        "goal_continuation_ref": None,
        "reason": "The model selected the advertised task control.",
    }


def _module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as exc:
        pytest.fail(f"planned action owner is unavailable: {name}: {exc}")




def test_evaluator_accepts_only_advertised_typed_task_control() -> None:
    """Evaluator validates typed output without classifying user wording."""

    evaluator_module = _module("kazusa_ai_chatbot.action_spec.evaluator")
    evaluator = evaluator_module.ActionSpecEvaluator(
        capabilities=_module(
            "kazusa_ai_chatbot.action_spec.registry",
        ).build_initial_action_capabilities(),
    )
    accepted = evaluator.evaluate(_action_spec())
    assert accepted["ok"] is True
    assert accepted["action_spec"]["params"]["control"] == _control()
    invalid = {**_action_spec(), "params": {"control": {
        **_control("cancel"),
        "unexpected": "closed",
    }}}
    rejected = evaluator.evaluate(invalid)
    assert rejected["ok"] is False


@pytest.mark.asyncio
async def test_control_claims_advertised_followup_or_cancels_without_interpreting_user_text() -> None:
    """Execution uses only the typed control and advertised affordance."""

    execution = _module("kazusa_ai_chatbot.action_spec.execution")
    handler = getattr(execution, "execute_accepted_task_control", None)
    if not callable(handler):
        pytest.fail("action execution owner lacks execute_accepted_task_control")
    lifecycle = AsyncMock()
    result = await handler(
        control={
            **_control(),
            "instruction": "Continue; the quoted word cancel is semantic task text.",
        },
        affordance=_affordance(),
        action_attempt_id="attempt-1",
        lifecycle=lifecycle,
    )
    assert result["status"] in {"queued", "claimed", "already_claimed"}
    lifecycle.assert_awaited_once()




@pytest.mark.asyncio
async def test_handler_binds_control_to_trusted_scope_and_task_affordance() -> None:
    """The handler rejects an unadvertised ref and forwards the valid one."""

    handlers = _module("kazusa_ai_chatbot.action_spec.handlers.background_work")
    handle = getattr(handlers, "handle_accepted_task_control", None)
    if not callable(handle):
        pytest.fail("background action handler lacks handle_accepted_task_control")
    claim = AsyncMock(return_value={"accepted_task_id": "task-1"})
    result = await handle(
        control=_control(),
        affordance=_affordance(),
        trusted_scope={
            "platform": "debug",
            "channel_id": "channel-1",
            "requester_global_user_id": "user-1",
        },
        claim_followup=claim,
    )
    assert result["accepted_task_id"] == "task-1"
    claim.assert_awaited_once()
    with pytest.raises(ValueError):
        await handle(
            control={**_control(), "accepted_task_ref": "accepted_task:other"},
            affordance=_affordance(),
            trusted_scope={
                "platform": "debug",
                "channel_id": "channel-1",
                "requester_global_user_id": "user-1",
            },
            claim_followup=claim,
        )


