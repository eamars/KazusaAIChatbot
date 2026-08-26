"""Production integration checks for retained local-context ownership."""

from __future__ import annotations

from typing import Any

import pytest

from kazusa_ai_chatbot.cognition_resolver import capabilities
from kazusa_ai_chatbot.time_boundary import build_turn_clock
from tests.cognition_test_helpers import canonical_user_message_episode

pytestmark = pytest.mark.asyncio


def _persona_state() -> dict[str, Any]:
    """Build a production-like state for the first-cycle prewarm boundary."""

    turn_clock = build_turn_clock("2026-07-04 09:30:00")
    episode = canonical_user_message_episode(
        episode_id="local-context-cutover-episode",
        percept_id="local-context-cutover-percept",
        storage_timestamp_utc=turn_clock["storage_timestamp_utc"],
        local_time_context=turn_clock["local_time_context"],
        user_input="Please check the local evidence.",
        platform="debug",
        platform_channel_id="channel-123",
        channel_type="private",
        platform_message_id="message-123",
        platform_user_id="platform-user-123",
        global_user_id="global-user-123",
        user_name="Test User",
        active_turn_platform_message_ids=["message-123"],
        active_turn_conversation_row_ids=["row-123"],
        debug_modes={},
        target_addressed_user_ids=["character-123"],
        target_broadcast=False,
    )
    return {
        "cognitive_episode": episode,
        "decontextualized_input": "Please check the local evidence.",
        "referents": [],
        "character_profile": {
            "name": "Kazusa",
            "global_user_id": "character-123",
        },
        "platform": "debug",
        "platform_channel_id": "channel-123",
        "channel_type": "private",
        "platform_message_id": "message-123",
        "platform_bot_id": "bot-123",
        "global_user_id": "global-user-123",
        "user_name": "Test User",
        "user_profile": {"relationship_state": 500},
        "storage_timestamp_utc": turn_clock["storage_timestamp_utc"],
        "local_time_context": turn_clock["local_time_context"],
        "prompt_message_context": {
            "body_text": "Please check the local evidence.",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-123"],
            "broadcast": False,
        },
        "channel_topic": "debug",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
        "conversation_progress": {"current_thread": "local evidence check"},
        "conversation_episode_state": None,
        "promoted_reflection_context": None,
    }


async def test_first_cycle_prewarm_uses_memory_worker_without_full_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First-cycle prewarm stays outside the full local-context resolver."""

    calls: list[dict[str, Any]] = []

    async def resolve_local_context(
        _request: dict[str, Any],
        _context: dict[str, Any],
        _options: dict[str, Any],
    ) -> dict[str, Any]:
        raise AssertionError("prewarm must not call resolve_local_context")

    class FakePersistentMemorySearchAgent:
        """Capture the direct shared-memory prewarm worker call."""

        async def run(
            self,
            task: str,
            context: dict[str, Any],
            max_attempts: int = 3,
        ) -> dict[str, Any]:
            calls.append({
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            })
            return {
                "resolved": True,
                "result": [
                    {
                        "content": "Shared memory evidence from prewarm.",
                        "memory_unit_id": "local-context-shared-unit",
                        "memory_name": "fact",
                        "memory_type": "fact",
                        "source_kind": "conversation_extracted",
                        "source_global_user_id": "",
                        "authority": "conversation_accepted",
                        "status": "active",
                        "scope_type": "global",
                        "privacy_review": {
                            "global_applicability": "global",
                            "target_specific_meaning_removed": True,
                            "affects_identity_or_boundaries": False,
                            "private_detail_risk": "low",
                            "user_details_removed": True,
                            "boundary_assessment": "deidentified global meaning",
                            "reviewer": "automated_llm",
                        },
                    },
                    {
                        "content": "Scoped user memory must not enter prewarm.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "global-user-123",
                    },
                ],
                "attempts": 1,
            }

    monkeypatch.setattr(
        capabilities,
        "resolve_local_context",
        resolve_local_context,
        raising=False,
    )
    monkeypatch.setattr(
        capabilities,
        "PersistentMemorySearchAgent",
        FakePersistentMemorySearchAgent,
        raising=False,
    )

    outcome = await capabilities.run_first_cycle_shared_memory_prewarm(
        _persona_state(),
    )

    assert outcome["status"] == "completed"
    assert outcome["reason_code"] == "shared_memory_ready"
    rag_result = outcome["rag_result"]
    assert rag_result["answer"] == ""
    assert "Shared memory evidence from prewarm." in repr(
        rag_result["memory_evidence"]
    )
    assert rag_result["user_memory_unit_candidates"] == []
    assert "Scoped user memory" not in repr(rag_result)
    assert len(calls) == 1
    worker_call = calls[0]
    assert worker_call["task"] == "Please check the local evidence."
    assert worker_call["max_attempts"] == 1
    assert worker_call["context"]["prompt_message_context"]["body_text"] == (
        "Please check the local evidence."
    )
