"""Recorder boundary-profile tests for conversation progress."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from kazusa_ai_chatbot.conversation_progress import recorder
from kazusa_ai_chatbot.conversation_progress import runtime
from kazusa_ai_chatbot.conversation_progress.models import (
    ConversationProgressRecordInput,
    ConversationProgressScope,
)
from kazusa_ai_chatbot.nodes.boundary_profile import (
    get_boundary_recovery_description,
    get_relationship_priority_description,
    get_self_integrity_description,
)


_BOUNDARY_PROFILE = {
    "self_integrity": 0.82,
    "control_sensitivity": 0.3,
    "compliance_strategy": "comply",
    "relational_override": 0.24,
    "control_intimacy_misread": 0.2,
    "boundary_recovery": "rebound",
    "authority_skepticism": 0.35,
}

_VALID_RECORDER_OUTPUT = {
    "continuity": "same_episode",
    "status": "active",
    "episode_label": "clarified_reference",
    "conversation_mode": "casual_chat",
    "episode_phase": "resolving",
    "topic_momentum": "stable",
    "current_thread": "clarified referent",
    "user_goal": "",
    "current_blocker": "",
    "user_state_updates": [],
    "assistant_moves": ["clarified referent"],
    "overused_moves": [],
    "open_loops": [],
    "resolved_threads": ["referent clarified"],
    "avoid_reopening": [],
    "emotional_trajectory": "settled",
    "next_affordances": ["continue normally"],
    "progression_guidance": "continue without carrying old suspicion",
}


class _FakeResponse:
    """Small LLM response stand-in."""

    def __init__(self, payload: dict):
        self.content = json.dumps(payload)


class _CapturingLLM:
    """Capture recorder messages while returning a fixed JSON payload."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.messages = []

    async def ainvoke(self, messages):
        self.messages = messages
        response = _FakeResponse(self.payload)
        return response


def _record_input(
    boundary_profile: dict,
) -> ConversationProgressRecordInput:
    """Build a recorder input fixture.

    Args:
        boundary_profile: Boundary configuration to include.

    Returns:
        Recorder input fixture.
    """

    record_input: ConversationProgressRecordInput = {
        "scope": ConversationProgressScope("qq", "channel-1", "user-1"),
        "timestamp": "2026-05-01T04:00:00+00:00",
        "prior_episode_state": None,
        "decontexualized_input": "I meant the other thing.",
        "chat_history_recent": [],
        "content_anchors": ["[DECISION] accept the clarification"],
        "logical_stance": "CONFIRM",
        "character_intent": "PROVIDE",
        "final_dialog": ["Got it, then I will use that meaning."],
        "boundary_profile": boundary_profile,
    }
    return record_input


@pytest.mark.asyncio
async def test_record_with_llm_sends_boundary_descriptors_not_config_values(monkeypatch) -> None:
    """Recorder prompt payload contains descriptors, not boundary config values."""

    fake_llm = _CapturingLLM(_VALID_RECORDER_OUTPUT)
    monkeypatch.setattr(recorder, "_recorder_llm", fake_llm)

    result = await recorder.record_with_llm(_record_input(_BOUNDARY_PROFILE))

    human_payload = json.loads(fake_llm.messages[1].content)
    profile_payload = human_payload["character_boundary_profile"]
    assert human_payload["current_turn_timestamp"] == "2026-05-01 16:00"
    assert profile_payload == {
        "boundary_recovery_description": get_boundary_recovery_description(
            _BOUNDARY_PROFILE["boundary_recovery"],
        ),
        "self_integrity_description": get_self_integrity_description(
            _BOUNDARY_PROFILE["self_integrity"],
        ),
        "relationship_priority_description": get_relationship_priority_description(
            _BOUNDARY_PROFILE["relational_override"],
        ),
    }
    assert "boundary_recovery" not in profile_payload
    assert "self_integrity" not in profile_payload
    assert "relational_override" not in profile_payload
    serialized_profile = json.dumps(profile_payload, ensure_ascii=False)
    assert "rebound" not in serialized_profile
    assert "0.82" not in serialized_profile
    assert "0.24" not in serialized_profile
    assert result["progression_guidance"] == (
        "continue without carrying old suspicion"
    )


@pytest.mark.asyncio
async def test_runtime_record_accepts_boundary_profile_without_schema_change(monkeypatch) -> None:
    """Runtime writes a normal episode document when boundary_profile is supplied."""

    recorder_callable = AsyncMock(return_value=dict(_VALID_RECORDER_OUTPUT))
    stored_documents = []

    def _store_completed_document(*, scope, document) -> None:
        stored_documents.append(document)

    monkeypatch.setattr(
        runtime.repository,
        "upsert_episode_state_guarded",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        runtime.cache,
        "store_completed_document",
        _store_completed_document,
    )
    progress_runtime = runtime.ConversationProgressRuntime(
        recorder_callable=recorder_callable,
    )

    result = await progress_runtime.record_turn_progress(
        record_input=_record_input(_BOUNDARY_PROFILE),
    )

    recorder_callable.assert_awaited_once()
    assert recorder_callable.await_args.args[0]["boundary_profile"] == (
        _BOUNDARY_PROFILE
    )
    assert result["written"] is True
    assert stored_documents[0]["next_affordances"] == ["continue normally"]
    assert "boundary_profile" not in stored_documents[0]
