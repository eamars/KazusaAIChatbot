"""Contract coverage for deterministic RAG3 subagent dispatch."""

import pytest

from kazusa_ai_chatbot.local_context_resolver.subagent import (
    dispatch_subagent_for_node,
)
from kazusa_ai_chatbot.local_context_resolver.subagent.media import MediaSubagent
from kazusa_ai_chatbot.local_context_resolver.subagent.source import (
    SourceEvidenceSubagent,
)


@pytest.mark.asyncio
async def test_dispatch_uses_memory_subagent_for_memory_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the bounded task envelope for a source-backed node."""

    captured: dict[str, object] = {}

    async def _fake_run(self, task, context, max_attempts):
        """Capture the deterministic task envelope without a database call."""

        del self, context, max_attempts
        captured.update(task)
        return {
            "schema_version": "local_context_subagent_result.v1",
            "resolved": False,
            "status": "unavailable",
            "result": {
                "source_records": [],
                "artifacts": [],
                "node_update": {},
            },
            "attempts": 0,
            "cache": {"enabled": False},
            "trace": {},
            "unresolved_items": [],
        }

    monkeypatch.setattr(SourceEvidenceSubagent, "run", _fake_run)
    result = await dispatch_subagent_for_node(
        active_node={
            "node_id": "task_1",
            "node_kind": "memory_evidence",
            "objective": "Find relevant remembered preferences.",
        },
        context={
            "character_name": "Character",
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
            "user_name": "User",
            "local_time_context": {},
            "conversation_progress": {},
            "chat_history_recent": [],
            "chat_history_wide": [],
        },
        dependency_context=[],
        max_attempts=1,
    )

    assert captured["subagent"] == "memory"
    assert captured["action"] == "collect_memory"
    assert result["status"] == "unavailable"


@pytest.mark.asyncio
async def test_dispatch_uses_recent_media_when_current_turn_has_no_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve a current-media node to the available recent image deterministically."""

    captured: dict[str, object] = {}

    async def _fake_run(self, task, context, max_attempts):
        """Capture the media task without running the image inspector."""

        del self, context, max_attempts
        captured.update(task)
        return {
            "schema_version": "local_context_subagent_result.v1",
            "resolved": False,
            "status": "unavailable",
            "result": {
                "source_records": [],
                "artifacts": [],
                "node_update": {},
            },
            "attempts": 0,
            "cache": {"enabled": False},
            "trace": {},
            "unresolved_items": [],
        }

    monkeypatch.setattr(MediaSubagent, "run", _fake_run)
    result = await dispatch_subagent_for_node(
        active_node={
            "node_id": "task_1",
            "node_kind": "current_turn_media",
            "objective": "Which shape is left of the white circle?",
        },
        context={
            "character_name": "Character",
            "platform": "debug",
            "platform_channel_id": "channel-1",
            "global_user_id": "user-1",
            "user_name": "User",
            "local_time_context": {},
            "conversation_progress": {},
            "chat_history_recent": [],
            "chat_history_wide": [],
            "session_media_refs": [{
                "turn_relation": "recent",
                "content_type": "image/png",
            }],
        },
        dependency_context=[],
        max_attempts=1,
    )

    task_payload = captured["payload"]
    assert isinstance(task_payload, dict)
    assert task_payload["selector"] == {
        "schema_version": "local_context_media_selector.v1",
        "selector_kind": "recent",
        "alias": None,
        "ordinal": 1,
        "question": "Which shape is left of the white circle?",
    }
    assert result["status"] == "unavailable"
