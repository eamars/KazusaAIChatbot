"""Executable tests for cognition-to-DSH capability projection."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tests.task_resolution_test_helpers import (
    _context,
    _goal_continuation_ref,
    _resolution_ref,
)


def _scene() -> dict[str, object]:
    return {
        "channel_scope": "private",
        "character_role": "Test Character",
        "current_user_role": "Test User",
        "semantic_scene": "A bounded task-resolution scene.",
        "public_group_scene": "",
        "conversation_continuity": "One continuing goal.",
        "semantic_temporal_context": "The test turn is current.",
    }


def _state() -> dict[str, object]:
    return {
        "character_profile": {"name": "Test Character"},
        "platform": "debug",
        "platform_channel_id": "channel-1",
        "channel_type": "private",
        "global_user_id": "user-1",
        "platform_user_id": "platform-user-1",
        "platform_message_id": "message-1",
        "platform_bot_id": "bot-1",
        "user_name": "Test User",
        "storage_timestamp_utc": "2026-08-30T00:00:00Z",
        "local_time_context": {"local_time": "2026-08-30 12:00"},
        "prompt_message_context": {"text": "Resolve this bounded goal."},
        "chat_history_recent": [],
        "chat_history_wide": [],
        "cognitive_episode": {
            "trigger_source": "user_message",
            "episode_id": "episode-task-001",
        },
        "cognition_scene_context": _scene(),
        "conversation_progress": {},
        "decontextualized_input": "Resolve this bounded goal.",
        "rag_result": {},
        "referents": [],
    }


def _request() -> dict[str, object]:
    return {
        "schema_version": "resolver_capability_request.v1",
        "capability_kind": "task_resolution_request",
        "objective": "Resolve this bounded goal.",
        "reason": "The goal requires DSH evidence.",
        "priority": "now",
        "goal_continuation_ref": _goal_continuation_ref(),
    }






def _result(status: str) -> dict[str, object]:
    continuation = _goal_continuation_ref()
    evidence = [{
        "schema_version": "task_resolution_evidence.v1",
        "evidence_id": "evidence-1",
        "task_node_id": "node-1",
        "specialist": "dsh",
        "summary": "semantic-ref-1",
        "provenance_refs": ["receipt-1", "sha256:content"],
        "limitations": [],
    }]
    remaining = [] if status == "resolved" else ["one bounded follow-up"]
    return {
        "schema_version": "task_resolution_result.v1",
        "semantic_objective": "Resolve this bounded goal.",
        "status": status,
        "scene_context": _scene(),
        "goal_continuation_ref": continuation,
        "evidence_state": "complete" if status == "resolved" else "partial",
        "evidence_excerpts": ['{"answer":"bounded"}'],
        "evidence_handles": ["semantic-ref-1"],
        "prompt_safe_summary": "The bounded goal has a typed result.",
        "evidence": evidence,
        "completed_subgoals": ["bounded evidence"],
        "remaining_needs": remaining,
        "checkpoint": {},
        "coding_run_context": {},
    }


def test_retried_task_projects_original_objective_and_completed_evidence() -> None:
    """A paraphrased retry retains the executed task's evidence and identity."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    request = {**_request(), "objective": "Reword the same bounded goal."}
    result = _result("resolved")

    observation = owner._task_resolution_observation(
        request,
        _state(),
        result,
        durably_promoted=False,
    )

    assert observation["status"] == "succeeded"
    assert observation["request_objective"] == result["semantic_objective"]
    assert observation["goal_continuation_ref"] == request["goal_continuation_ref"]
    assert observation["knowledge_projection"]["knowledge_we_know_so_far"] == (
        result["evidence_excerpts"]
    )
    assert request["objective"] == "Reword the same bounded goal."


def test_task_result_from_another_goal_is_rejected() -> None:
    """Matching prose cannot substitute for the caller's trusted goal identity."""

    from kazusa_ai_chatbot.cognition_episode import build_goal_continuation_ref
    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner
    from kazusa_ai_chatbot.cognition_resolver.contracts import ResolverValidationError

    result = _result("resolved")
    result["goal_continuation_ref"] = build_goal_continuation_ref(
        source_episode_id="another-episode",
        source_message_id="another-message",
        branch_id="b1",
        goal_ref={"scope": "user", "kind": "goal", "entity_id": "another-goal"},
    )

    with pytest.raises(ResolverValidationError, match="continuation reference"):
        owner._task_resolution_observation(
            _request(),
            _state(),
            result,
            durably_promoted=False,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("binding_state", ["active", "terminal"])
async def test_cognition_retry_preserves_promoted_task_and_delivery(binding_state: str) -> None:
    """Replaying an admitted checkpoint performs no new work or promotion."""

    from kazusa_ai_chatbot.task_resolution import service as owner

    context = _context()
    request = {**_request(), "objective": "A regenerated description of the same goal."}
    reference = _resolution_ref()
    binding = {
        "state": binding_state,
        "semantic_objective": "The originally accepted goal.",
        "goal_continuation_ref": context["goal_continuation_ref"],
        "source_scope": owner._source_scope(context),
        "start_spec": owner._build_start_spec(request, context),
        "resolution_ref": reference,
        "current_accepted_task_id": "existing-task",
        "current_background_work_job_id": "existing-job",
    }
    store = SimpleNamespace(
        find_binding_by_session=AsyncMock(return_value=binding),
        create_task_binding=AsyncMock(),
    )
    runtime = SimpleNamespace(open=AsyncMock())
    accepted = SimpleNamespace(create_or_return_active_accepted_task=AsyncMock())
    queue = SimpleNamespace(enqueue=AsyncMock())

    replay = await owner.resolve_task_inline(
        request, context, inline_budget_seconds=1, runtime=runtime, binding_store=store,
    )
    promoted = await owner.promote_deferred_task_resolution(
        replay, context, binding_store=store, accepted_task_store=accepted,
        background_queue=queue,
    )

    assert promoted["status"] == "deferred"
    assert promoted["semantic_objective"] == "The originally accepted goal."
    assert promoted["checkpoint"] == reference
    runtime.open.assert_not_awaited()
    store.create_task_binding.assert_not_awaited()
    accepted.create_or_return_active_accepted_task.assert_not_awaited()
    queue.enqueue.assert_not_awaited()








def test_task_resolution_v2_context_projects_trusted_source_and_original_episode_ref(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real context projector carries trusted source and original episode identity."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    monkeypatch.setattr(owner, "list_session_media_refs", lambda _scope: [])
    context = owner._task_resolution_execution_context_from_state(
        _state(),
        goal_continuation_ref=_goal_continuation_ref(),
    )

    assert context["schema_version"] == "task_resolution_execution_context.v2"
    assert context["requester_display_name"] == "Test User"
    assert context["source_platform_bot_id"] == "bot-1"
    assert context["source_trigger_source"] == "user_message"
    assert context["source_llm_trace_id"] == ""
    assert context["brain_conversation_ref"] == "episode-task-001"
    assert "coding_workspace_root" not in context









