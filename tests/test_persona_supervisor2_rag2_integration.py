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
                    "raw_result": {
                        "global_user_id": "user-1",
                        "user_memory_context": {
                            "objective_facts": [
                                {
                                    "fact": "User likes tea",
                                    "subjective_appraisal": "Kazusa sees this as a stable preference.",
                                    "relationship_signal": "Offer tea-related continuity.",
                                }
                            ]
                        },
                    },
                }
            ],
            "unknown_slots": [],
            "loop_count": 1,
        }

    monkeypatch.setattr(supervisor_module, "call_rag_supervisor", _call_rag_supervisor)

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "你记得我喜欢什么吗？",
        "referents": [],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    })

    assert captured["original_query"] == "你记得我喜欢什么吗？"
    assert captured["character_name"] == "Kazusa"
    assert captured["context"]["channel_type"] == "group"
    assert captured["context"]["message_envelope"]["body_text"] == "clean body"
    assert captured["context"]["chat_history_recent"] == []
    assert result["rag_result"]["answer"] == "resolved"
    assert result["rag_result"]["user_image"]["user_memory_context"]["objective_facts"][0]["fact"] == "User likes tea"


@pytest.mark.asyncio
async def test_stage_1_research_skips_rag_for_unresolved_referents(monkeypatch) -> None:
    """Unresolved required references should skip RAG and preserve payload shape."""
    called = False

    async def _call_rag_supervisor(*, original_query: str, character_name: str, context: dict) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(supervisor_module, "call_rag_supervisor", _call_rag_supervisor)

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"},
        ],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    })

    rag_result = result["rag_result"]
    assert called is False
    assert rag_result["answer"] == ""
    assert rag_result["user_image"]["user_memory_context"]["stable_patterns"] == []
    assert rag_result["user_image"]["user_memory_context"]["active_commitments"] == []
    assert rag_result["character_image"] == {}
    assert rag_result["memory_evidence"] == []
    assert rag_result["conversation_evidence"] == []
    assert rag_result["external_evidence"] == []
    assert rag_result["supervisor_trace"]["unknown_slots"] == []


@pytest.mark.asyncio
async def test_stage_1_research_runs_rag_for_mixed_referents(monkeypatch) -> None:
    """Mixed referents should not trigger the old binary RAG skip cliff."""
    called = False

    async def _call_rag_supervisor(*, original_query: str, character_name: str, context: dict) -> dict:
        nonlocal called
        called = True
        return {
            "answer": "partial evidence",
            "known_facts": [],
            "unknown_slots": [],
            "loop_count": 1,
        }

    monkeypatch.setattr(supervisor_module, "call_rag_supervisor", _call_rag_supervisor)

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "他上次说的那些关于X的话是什么意思？",
        "referents": [
            {"phrase": "他", "referent_role": "subject", "status": "resolved"},
            {"phrase": "那些话", "referent_role": "object", "status": "unresolved"},
        ],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    })

    assert called is True
    assert result["rag_result"]["answer"] == "partial evidence"


@pytest.mark.asyncio
async def test_stage_1_research_skips_when_referents_are_all_unresolved(monkeypatch) -> None:
    """Structured unresolved referents should be authoritative."""
    called = False

    async def _call_rag_supervisor(*, original_query: str, character_name: str, context: dict) -> dict:
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(supervisor_module, "call_rag_supervisor", _call_rag_supervisor)

    result = await supervisor_module.stage_1_research({
        "decontexualized_input": "这些是什么意思？",
        "referents": [
            {"phrase": "这些", "referent_role": "object", "status": "unresolved"},
        ],
        "character_profile": {"name": "Kazusa", "global_user_id": "character-1"},
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "channel_type": "group",
        "platform_bot_id": "bot-1",
        "global_user_id": "user-1",
        "user_name": "User",
        "user_profile": {"affinity": 500},
        "timestamp": "2026-04-27T00:00:00+12:00",
        "message_envelope": {
            "body_text": "clean body",
            "raw_wire_text": "clean body",
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": ["character-1"],
            "broadcast": False,
        },
        "channel_topic": "test",
        "chat_history_recent": [],
        "chat_history_wide": [],
        "reply_context": {},
        "indirect_speech_context": "",
    })

    assert called is False
    assert result["rag_result"]["answer"] == ""
