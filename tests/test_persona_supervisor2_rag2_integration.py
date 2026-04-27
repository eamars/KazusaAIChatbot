"""Tests for persona supervisor RAG2 stage wiring."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2 as supervisor_module


@pytest.mark.asyncio
async def test_stage_1_research_calls_rag2_and_projects_payload(monkeypatch) -> None:
    captured: dict = {}

    async def _call_rag_supervisor(*, original_query: str, character_name: str, context: dict) -> dict:
        captured["original_query"] = original_query
        captured["character_name"] = character_name
        captured["context"] = context
        return {
            "answer": "resolved",
            "known_facts": [
                {
                    "slot": "profile",
                    "agent": "user_profile_agent",
                    "resolved": True,
                    "summary": "profile",
                    "raw_result": {"global_user_id": "user-1", "objective_facts": [{"fact": "User likes tea"}]},
                }
            ],
            "unknown_slots": [],
            "loop_count": 1,
        }

    monkeypatch.setattr(supervisor_module, "call_rag_supervisor", _call_rag_supervisor)

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "你记得我喜欢什么吗？",
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    })

    assert captured["original_query"] == "你记得我喜欢什么吗？"
    assert captured["character_name"] == "Kazusa"
    assert captured["context"]["channel_type"] == "group"
    assert captured["context"]["chat_history_recent"] == []
    assert result["rag_result"]["answer"] == "resolved"
    assert result["rag_result"]["user_image"]["objective_facts"][0]["fact"] == "User likes tea"
