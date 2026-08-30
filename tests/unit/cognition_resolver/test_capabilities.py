"""Executable tests for cognition-to-DSH capability projection."""

from __future__ import annotations

import pytest

from tests.task_resolution_test_helpers import _goal_continuation_ref


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


def _dsh_resolution_ref() -> dict[str, object]:
    """Build the opaque DSH reference carried by deferred results."""

    return {
        "schema_version": "dsh_resolution_ref.v1",
        "resolution_thread_id": "thread-1",
        "segment_id": "segment-1",
        "dsh_session_id": "session-1",
        "activation_id": "activation-1",
        "lease_epoch": 1,
        "document_revision": 0,
        "last_committed_seq": 0,
    }


def _admission() -> dict[str, object]:
    """Build the transient identity returned before a DSH claim."""

    return {
        "schema_version": "task_resolution_admission.v1",
        "accepted_task_id": "task-1",
        "background_work_job_id": "job-1",
        "task_session_id": "session-1",
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


def test_task_resolution_preserves_recurrence_and_maps_dsh_deferred_result() -> None:
    """Real result projection preserves statuses and DSH evidence lineage."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    for status, expected_state in (("resolved", "complete"), ("partial", "partial")):
        observation = owner._task_resolution_observation(
            _request(),
            _state(),
            _result(status),
            durably_promoted=False,
        )
        assert observation["status"] == "succeeded"
        assert observation["task_resolution_evidence_state"]["state"] == expected_state
        assert observation["evidence_refs"][0]["owner"] == "dsh"
        assert observation["goal_continuation_ref"] == _request()["goal_continuation_ref"]

    deferred = _result("resolved")
    deferred.update({
        "status": "deferred",
        "evidence_state": "pending",
        "evidence": [],
        "evidence_excerpts": [],
        "evidence_handles": [],
        "completed_subgoals": [],
        "remaining_needs": ["continue the bounded goal"],
        "checkpoint": _dsh_resolution_ref(),
    })
    observation = owner._task_resolution_observation(
        _request(),
        _state(),
        deferred,
        durably_promoted=True,
    )
    assert observation["status"] == "succeeded"
    assert observation["task_resolution_evidence_state"]["state"] == "pending"
    assert observation["evidence_refs"] == []

    admission = owner._task_resolution_admission_observation(
        {**_request(), "priority": "background"},
        _state(),
        _admission(),
    )
    assert admission["status"] == "succeeded"
    assert admission["task_resolution_evidence_state"]["state"] == "pending"
    assert admission["evidence_refs"] == []
    assert "accepted_task_id" not in admission
    assert "background_work_job_id" not in admission
    assert "task_session_id" not in admission


def test_task_resolution_evidence_refs_ignore_unprojectable_artifact_handles() -> None:
    """Artifact handles may accompany factual evidence without a second ref."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    result = _result("resolved")
    result["evidence_handles"].append("artifact-1")
    observation = owner._task_resolution_observation(
        _request(),
        _state(),
        result,
        durably_promoted=False,
    )

    assert len(observation["evidence_refs"]) == 1
    assert observation["evidence_refs"][0]["evidence_id"] == "semantic-ref-1"


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


def test_task_resolution_context_requires_current_episode_identity() -> None:
    """The DSH lineage carrier fails closed without a trusted episode id."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    state = _state()
    state.pop("brain_conversation_ref", None)
    state.pop("conversation_ref", None)
    state["cognitive_episode"] = {"trigger_source": "user_message"}

    with pytest.raises(
        owner.ResolverValidationError,
        match="cognitive_episode.episode_id",
    ):
        owner._task_resolution_execution_context_from_state(
            state,
            goal_continuation_ref=_goal_continuation_ref(),
        )


def test_task_resolution_context_uses_episode_identity_over_stale_state_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The current cognitive episode exclusively owns DSH conversation identity."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    monkeypatch.setattr(owner, "list_session_media_refs", lambda _scope: [])
    state = _state()
    state["brain_conversation_ref"] = "stale-brain-ref"
    state["conversation_ref"] = "stale-conversation-ref"
    state["cognitive_episode"] = {
        "trigger_source": "user_message",
        "episode_id": "current-episode-ref",
    }
    context = owner._task_resolution_execution_context_from_state(
        state,
        goal_continuation_ref=_goal_continuation_ref(),
    )

    assert context["brain_conversation_ref"] == "current-episode-ref"


def test_task_capability_uses_runtime_readiness_without_legacy_fallback() -> None:
    """Admission invokes the real readiness validator and fails closed."""

    from kazusa_ai_chatbot.cognition_resolver import capabilities as owner

    owner.validate_task_resolution_execution_readiness(
        _state(),
        cognition_scene_context=_scene(),
    )
    invalid_state = {**_state(), "platform_message_id": ""}
    with pytest.raises(ValueError, match="platform_message_id"):
        owner.validate_task_resolution_execution_readiness(
            invalid_state,
            cognition_scene_context=_scene(),
        )
    with pytest.raises(ValueError):
        owner.validate_task_resolution_execution_readiness(
            _state(),
            cognition_scene_context={**_scene(), "private": "hidden"},
        )
