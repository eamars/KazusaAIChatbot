"""Deterministic contract tests for RAG2 top-level capability agents."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from kazusa_ai_chatbot.rag.conversation_evidence import ConversationEvidenceAgent
from kazusa_ai_chatbot.rag.conversation_evidence import (
    selector as conversation_evidence_module,
)
from kazusa_ai_chatbot.rag.evidence_coverage import (
    assess_evidence_coverage,
    requested_coverage_items,
)
from kazusa_ai_chatbot.rag.live_context import LiveContextAgent
from kazusa_ai_chatbot.rag.memory_evidence import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.memory_evidence import selector as memory_evidence_module
from kazusa_ai_chatbot.rag.person_context import PersonContextAgent


class _FakeWorker:
    """Small async worker test double that records helper-agent calls."""

    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls: list[dict] = []

    async def run(
        self,
        task: str,
        context: dict,
        max_attempts: int = 3,
    ) -> dict:
        """Record the helper invocation and return the configured result."""
        self.calls.append(
            {
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            }
        )
        return_value = self.result
        return return_value


def _base_context(**overrides: object) -> dict:
    """Build the scoped context needed by standalone capability tests."""
    context = {
        "platform": "qq",
        "platform_channel_id": "chan-1",
        "platform_user_id": "platform-user-1",
        "global_user_id": "user-1",
        "user_name": "Tester",
        "current_timestamp_utc": "2026-05-02T00:00:00+00:00",
        "known_facts": [],
        "current_slot": "slot 1",
    }
    context.update(overrides)
    return context


def _web_result(text: str) -> dict:
    """Build a resolved web worker payload."""
    result = {
        "resolved": True,
        "result": text,
        "attempts": 1,
        "cache": {
            "enabled": False,
            "hit": False,
            "cache_name": "",
            "reason": "agent_not_cacheable",
        },
    }
    return result


@pytest.mark.asyncio
async def test_live_context_explicit_location_goes_directly_to_web() -> None:
    """Explicit live targets should not spend a worker call on target lookup."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(
        _web_result(
            "Auckland is 17 C now. Source: https://weather.example/auckland"
        )
    )
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer current temperature for explicit location Auckland",
        _base_context(),
    )

    assert result["resolved"] is True
    assert result["attempts"] == 1
    assert result["cache"]["enabled"] is False
    assert result["cache"]["reason"] == "capability_orchestrator_uncached"
    assert len(web_worker.calls) == 1
    assert memory_worker.calls == []
    assert conversation_worker.calls == []
    assert "Auckland" in web_worker.calls[0]["task"]

    payload = result["result"]
    assert payload["capability"] == "live_context"
    assert payload["primary_worker"] == "web_agent3"
    assert payload["supporting_workers"] == []
    assert payload["projection_payload"] == {
        "external_text": "Auckland is 17 C now. Source: https://weather.example/auckland",
        "url": "https://weather.example/auckland",
    }
    assert payload["resolved_refs"] == [
        {
            "ref_type": "location",
            "role": "target_location",
            "text": "Auckland",
        }
    ]


@pytest.mark.asyncio
async def test_live_context_character_location_uses_memory_only_for_target_scope() -> None:
    """Character-local live facts may use memory only to resolve location."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("123 Example Street weather is mild."))
    memory_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "content": "The active character's official address is 123 Example Street."
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker

    result = await agent.run(
        "Live-context: answer current weather for the active character's location",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(memory_worker.calls) == 1
    assert "target_scope_lookup" in memory_worker.calls[0]["task"]
    assert "do not retrieve live" in memory_worker.calls[0]["task"]
    assert len(web_worker.calls) == 1

    payload = result["result"]
    assert payload["primary_worker"] == "web_agent3"
    assert payload["supporting_workers"] == ["persistent_memory_search_agent"]
    assert "target_scope_lookup" in payload["source_policy"]
    assert payload["resolved_refs"][0]["role"] == "character_default"
    assert "123 Example Street" in payload["resolved_refs"][0]["text"]


@pytest.mark.asyncio
async def test_live_context_user_location_uses_recent_conversation_scope() -> None:
    """User-local live facts should use recent same-user conversation scope."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("Christchurch is 12 C now."))
    conversation_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "I am in Christchurch this morning.",
                    "global_user_id": "user-1",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.conversation_search_agent = conversation_worker
    agent.memory_search_agent = memory_worker

    result = await agent.run(
        "Live-context: answer current temperature for the current user's location if recently stated",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(conversation_worker.calls) == 1
    assert conversation_worker.calls[0]["context"]["global_user_id"] == "user-1"
    assert "target_scope_lookup" in conversation_worker.calls[0]["task"]
    assert memory_worker.calls == []
    assert len(web_worker.calls) == 1
    assert "Christchurch" in web_worker.calls[0]["task"]

    payload = result["result"]
    assert payload["supporting_workers"] == ["conversation_search_agent"]
    assert payload["resolved_refs"][0]["role"] == "user_recent"


@pytest.mark.asyncio
async def test_live_context_user_location_refuses_without_fallback() -> None:
    """Missing user-local target must not fall back to character memory."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("should not be called"))
    conversation_worker = _FakeWorker(
        {
            "resolved": False,
            "result": [],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.conversation_search_agent = conversation_worker
    agent.memory_search_agent = memory_worker

    result = await agent.run(
        "Live-context: answer current temperature for the current user's location if recently stated",
        _base_context(),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["location"]
    assert "user_recent" in result["result"]["source_policy"]
    assert len(conversation_worker.calls) == 1
    assert memory_worker.calls == []
    assert web_worker.calls == []


@pytest.mark.asyncio
async def test_live_context_opening_status_explicit_target() -> None:
    """Opening status is a live external fact with an explicit public target."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("Christchurch Adventure Park is open."))
    agent.web_agent = web_worker

    result = await agent.run(
        "Live-context: answer current opening status for Christchurch Adventure Park this weekend",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(web_worker.calls) == 1
    assert "opening_status" in web_worker.calls[0]["task"]
    assert "Christchurch Adventure Park this weekend" in web_worker.calls[0]["task"]
    assert result["result"]["evidence"] == ["Christchurch Adventure Park is open."]


@pytest.mark.asyncio
async def test_live_context_current_local_time_uses_runtime_state_only() -> None:
    """Current character-local time should resolve from sanitized runtime state."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("should not be called"))
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer active character current local time",
        _base_context(
            local_time_context={
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            }
        ),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_worker"] == "runtime_context_provider"
    assert result["result"]["supporting_workers"] == []
    assert result["result"]["worker_payloads"]["runtime_context_provider"] == {
        "current_local_datetime": "2026-05-03 14:53",
        "current_local_weekday": "Sunday",
    }
    assert "2026-05-03 14:53" in result["result"]["selected_summary"]
    assert web_worker.calls == []
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_live_context_current_local_date_uses_runtime_state_only() -> None:
    """Current character-local date should resolve from sanitized runtime state."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("should not be called"))
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer active character current local date",
        _base_context(
            local_time_context={
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            }
        ),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_worker"] == "runtime_context_provider"
    assert result["result"]["supporting_workers"] == []
    assert "2026-05-03" in result["result"]["selected_summary"]
    assert web_worker.calls == []
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_live_context_current_local_weekday_uses_runtime_state_only() -> None:
    """Current character-local weekday should resolve from sanitized runtime state."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("should not be called"))
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer active character current local weekday",
        _base_context(
            local_time_context={
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            }
        ),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_worker"] == "runtime_context_provider"
    assert result["result"]["supporting_workers"] == []
    assert "Sunday" in result["result"]["selected_summary"]
    assert web_worker.calls == []
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_memory_evidence_selector_uses_current_user_scope_from_query(
    monkeypatch,
) -> None:
    """First-person durable recall should reach the selector LLM."""
    agent = MemoryEvidenceAgent()
    scoped_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "memory_rows": [
                    {
                        "content": "The user prefers reliable recall over latency.",
                        "source_system": "user_memory_units",
                    }
                ]
            },
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "test"},
        }
    )
    persistent_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [{"content": "World-level memory should not be used."}],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "test"},
        }
    )
    agent.user_memory_agent = scoped_worker
    agent.search_agent = persistent_worker
    selector_calls: list[list[object]] = []

    class _SelectorLLM:
        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            selector_calls.append(messages)
            response = SimpleNamespace(
                content=(
                    '{"worker": "user_memory_evidence_agent", '
                    '"reason": "scoped current-user continuity evidence"}'
                )
            )
            return response

    monkeypatch.setattr(memory_evidence_module, "_selector_llm", _SelectorLLM())

    result = await agent.run(
        "Memory-evidence: retrieve durable evidence about decision preference for architecture",
        _base_context(
            original_query=(
                "What did I care about most when choosing the architecture?"
            ),
        ),
    )

    assert result["resolved"] is True
    assert len(selector_calls) == 1
    assert len(scoped_worker.calls) == 1
    assert persistent_worker.calls == []
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"
    assert result["result"]["coverage"]["evidence_quality"] == "confirmed"
    assert result["result"]["confirmed_evidence"] == [
        "The user prefers reliable recall over latency."
    ]


@pytest.mark.asyncio
async def test_conversation_evidence_marks_partial_multi_target_result_unresolved() -> None:
    """A multi-target slot should not resolve from evidence for only one target."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                (
                    0.91,
                    {
                        "body_text": "Speaker: Model Alpha was quoted at 1200.",
                        "display_name": "Speaker",
                        "platform": "qq",
                        "platform_channel_id": "chan-1",
                    },
                )
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve price discussion for "
            "Model Alpha and Model Beta respectively speaker=any_speaker"
        ),
        _base_context(),
    )

    assert result["resolved"] is False
    assert len(search_worker.calls) == 1
    payload = result["result"]
    assert payload["coverage"]["evidence_quality"] == "partial"
    assert "Model Alpha" in payload["coverage"]["covered_items"]
    assert "Model Beta" in payload["coverage"]["missing_items"]
    assert payload["confirmed_evidence"] == []
    assert payload["partial_evidence"] == ["Speaker: Model Alpha was quoted at 1200."]
    assert payload["nearby_evidence"] == []
    assert payload["evidence"] == ["Speaker: Model Alpha was quoted at 1200."]
    assert payload["selected_summary"] == "Speaker: Model Alpha was quoted at 1200."


@pytest.mark.asyncio
async def test_conversation_evidence_resolves_covered_retrieval_candidate() -> None:
    """Covered retrieval evidence should not re-enter just because worker is unsure."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": False,
            "result": [
                {
                    "body_text": "没拧紧啊",
                    "display_name": "小钳子",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                },
                {
                    "body_text": "其实我发现有很简单的方法来处理耗材余量估算",
                    "display_name": "小钳子",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve messages around that timestamp "
            "speaker=any_speaker to find context mentioning '没拧紧' "
            "or '耗材余量估算'"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is True
    payload = result["result"]
    assert payload["missing_context"] == []
    assert payload["coverage"]["evidence_quality"] == "confirmed"
    assert payload["confirmed_evidence"] == [
        "小钳子: 没拧紧啊",
        "小钳子: 其实我发现有很简单的方法来处理耗材余量估算",
    ]
    assert payload["partial_evidence"] == []


@pytest.mark.asyncio
async def test_conversation_evidence_keeps_value_identification_unresolved() -> None:
    """Anchor hits alone should not satisfy slots asking to identify a value."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": False,
            "result": [
                {
                    "body_text": (
                        "<image>终端截图，包含红色错误信息 Too many retries。</image>"
                    ),
                    "display_name": "小钳子",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve messages containing "
            "'Too many retries' in QQ group 905393941 to identify the "
            "message timestamp"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is False
    payload = result["result"]
    assert payload["missing_context"] == ["conversation_evidence"]
    assert payload["coverage"]["evidence_quality"] == "partial"
    assert payload["confirmed_evidence"] == []
    assert payload["partial_evidence"] == [
        "小钳子: <image>终端截图，包含红色错误信息 Too many retries。</image>"
    ]


@pytest.mark.asyncio
async def test_conversation_evidence_relation_slot_requires_packet_relation() -> None:
    """A relation-dependent slot should not resolve from the seed row alone."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "后面这句提到了 Google Drive。",
                    "display_name": "Nightfall",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "seed",
                    "timestamp": "2026-05-22T09:10:00+00:00",
                    "methods": ["semantic"],
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve Google Drive discussion "
            "relation=previous_message speaker=any_speaker"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is False
    payload = result["result"]
    assert payload["missing_context"] == [
        "conversation_relation:previous_message"
    ]
    assert payload["projection_payload"]["packets"] == []


@pytest.mark.asyncio
async def test_conversation_evidence_relation_packet_reduces_seed_and_previous() -> None:
    """Relation packets should expose seed plus required nearby context."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "Google Drive 又不是第一次这样了。",
                    "display_name": "Nightfall",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "seed",
                    "timestamp": "2026-05-22T09:10:00+00:00",
                    "methods": ["semantic"],
                },
                {
                    "body_text": "<image>Google Drive 权限禁止的截图。</image>",
                    "display_name": "Nightfall",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "previous",
                    "timestamp": "2026-05-22T09:09:50+00:00",
                    "methods": ["neighbor"],
                    "relation_to_seed": "previous_message",
                    "seed_platform_message_id": "seed",
                    "seed_timestamp": "2026-05-22T09:10:00+00:00",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve Google Drive discussion "
            "relation=previous_message speaker=any_speaker"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is True
    payload = result["result"]
    assert payload["missing_context"] == []
    assert payload["projection_payload"]["packets"][0]["relation_types"] == [
        "previous_message"
    ]
    assert "命中消息" in payload["selected_summary"]
    assert "上一条" in payload["selected_summary"]
    assert "<image>Google Drive 权限禁止的截图。</image>" in payload["selected_summary"]


@pytest.mark.asyncio
async def test_conversation_evidence_relation_packet_keeps_direct_neighbor_seed() -> None:
    """A direct hit should remain a packet seed even if also tagged as nearby."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "Google Drive 又不是第一次这样了。",
                    "display_name": "Nightfall",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "seed",
                    "timestamp": "2026-05-22T09:10:00+00:00",
                    "methods": ["semantic", "neighbor"],
                    "relation_to_seed": "next_message",
                    "seed_platform_message_id": "previous",
                },
                {
                    "body_text": "<image>Google Drive 权限禁止的截图。</image>",
                    "display_name": "Nightfall",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "previous",
                    "timestamp": "2026-05-22T09:09:50+00:00",
                    "methods": ["semantic", "neighbor"],
                    "relation_to_seed": "previous_message",
                    "seed_platform_message_id": "seed",
                    "seed_timestamp": "2026-05-22T09:10:00+00:00",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve Google Drive discussion "
            "relation=previous_message speaker=any_speaker"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is True
    payload = result["result"]
    assert payload["missing_context"] == []
    assert payload["projection_payload"]["packets"][0]["relation_types"] == [
        "previous_message"
    ]


@pytest.mark.asyncio
async def test_conversation_evidence_non_relation_packets_require_keyword_seed() -> None:
    """Ordinary exact slots should not promote unrelated semantic packets."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "Google Drive 又不是第一次这样了。",
                    "display_name": "Nightfall",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "seed",
                    "timestamp": "2026-05-22T09:10:00+00:00",
                    "methods": ["semantic", "keyword:Google Drive"],
                },
                {
                    "body_text": "<image>Google Drive 账号封禁截图。</image>",
                    "display_name": "Nightfall",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "previous",
                    "timestamp": "2026-05-22T09:09:50+00:00",
                    "methods": ["neighbor"],
                    "relation_to_seed": "previous_message",
                    "seed_platform_message_id": "seed",
                    "seed_timestamp": "2026-05-22T09:10:00+00:00",
                },
                {
                    "body_text": "RAG 搜不到就不能作为事实基础。",
                    "display_name": "Tester",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "noise-seed",
                    "timestamp": "2026-05-22T09:20:00+00:00",
                    "methods": ["semantic"],
                },
                {
                    "body_text": "后台可以看到么",
                    "display_name": "Other",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                    "platform_message_id": "noise-relation",
                    "timestamp": "2026-05-22T09:19:50+00:00",
                    "methods": ["neighbor"],
                    "relation_to_seed": "previous_message",
                    "seed_platform_message_id": "noise-seed",
                    "seed_timestamp": "2026-05-22T09:20:00+00:00",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve messages containing exact phrase "
            "'Google Drive 又不是第一次这样了' speaker=any_speaker"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is True
    selected_summary = result["result"]["selected_summary"]
    assert "Google Drive 账号封禁截图" in selected_summary
    assert "RAG 搜不到" not in selected_summary
    assert "后台可以看到么" not in selected_summary
    assert len(result["result"]["projection_payload"]["packets"]) == 1


def test_conversation_evidence_media_relation_slot_uses_search_path() -> None:
    """Media-adjacent conversation slots should not be routed to memory."""

    plan = conversation_evidence_module._deterministic_plan(
        (
            "Conversation-evidence: retrieve message with attachment preceding "
            "the Google Drive quote to identify content of social media screenshot"
        )
    )

    assert plan is not None
    assert plan["worker"] == "conversation_search_agent"
    assert plan["reason"] == "relation or media-bearing conversation evidence"


@pytest.mark.asyncio
async def test_conversation_evidence_matches_cjk_target_spacing_variants() -> None:
    """CJK target coverage should tolerate common chat spelling variants."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "Nyan: 闪铸的c5那边多少钱",
                    "display_name": "Nyan",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                },
                {
                    "body_text": "蚝爹油: @Nyan 1800\nreply: 闪铸的c5那边多少钱",
                    "display_name": "蚝爹油",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve messages mentioning "
            "'闪铸 C5' in QQ group 905393941"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is True
    payload = result["result"]
    assert payload["coverage"]["requested_items"] == ["闪铸 C5"]
    assert payload["coverage"]["covered_items"] == ["闪铸 C5"]
    assert payload["coverage"]["evidence_quality"] == "confirmed"
    assert "Nyan: 闪铸的c5那边多少钱" in payload["confirmed_evidence"]


@pytest.mark.asyncio
async def test_conversation_price_evidence_requires_value_per_target() -> None:
    """Price tasks should not resolve from target mentions without values."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "Nyan: 闪铸的c5那边多少钱",
                    "display_name": "Nyan",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                },
                {
                    "body_text": "蚝爹油: X2D 是 $649 起",
                    "display_name": "蚝爹油",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve price discussion for "
            "'X2D' and '闪铸 C5' in QQ group 905393941"
        ),
        _base_context(platform_channel_id="905393941"),
    )

    assert result["resolved"] is False
    payload = result["result"]
    assert payload["coverage"]["evidence_quality"] == "partial"
    assert payload["coverage"]["covered_items"] == ["X2D"]
    assert payload["coverage"]["missing_items"] == ["闪铸 C5"]
    assert payload["partial_evidence"] == [
        "Nyan: 闪铸的c5那边多少钱",
        "蚝爹油: X2D 是 $649 起",
    ]


@pytest.mark.asyncio
async def test_conversation_continuation_carries_value_intent() -> None:
    """Narrowed follow-up slots should keep unresolved price intent."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "Nyan: 闪铸的c5那边多少钱",
                    "display_name": "Nyan",
                    "platform": "qq",
                    "platform_channel_id": "905393941",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    prior_fact = {
        "agent": "conversation_evidence_agent",
        "resolved": False,
        "slot": (
            "Conversation-evidence: retrieve price discussion for "
            "'X2D' and '闪铸 C5'"
        ),
        "raw_result": {
            "coverage": {
                "requested_items": ["X2D", "闪铸 C5"],
                "missing_items": ["闪铸 C5"],
            }
        },
    }
    result = await agent.run(
        "Conversation-evidence: retrieve messages mentioning '闪铸 C5'",
        _base_context(
            platform_channel_id="905393941",
            known_facts=[prior_fact],
        ),
    )

    assert result["resolved"] is False
    payload = result["result"]
    assert payload["coverage"]["evidence_quality"] == "nearby"
    assert payload["coverage"]["missing_items"] == ["闪铸 C5"]
    assert payload["nearby_evidence"] == ["Nyan: 闪铸的c5那边多少钱"]


def test_requested_coverage_items_ignores_command_scaffold() -> None:
    """Coverage targets should come from explicit entities, not command casing."""
    items = requested_coverage_items(
        (
            'Conversation-evidence: Find Retrieve Price Discussion for '
            '"Model Alpha" and "Model Beta" speaker=any_speaker'
        )
    )

    assert items == ["Model Alpha", "Model Beta"]


def test_requested_coverage_items_ignores_platform_scaffold() -> None:
    """Platform/channel labels should not become required evidence targets."""
    items = requested_coverage_items(
        (
            "Conversation-evidence: retrieve messages mentioning "
            "'闪铸 C5' in QQ group 905393941"
        )
    )

    assert items == ["闪铸 C5"]


def test_requested_coverage_items_narrows_specific_category_context() -> None:
    """Specificity markers should keep broad categories out of target coverage."""
    items = requested_coverage_items(
        (
            "Memory-evidence: retrieve durable evidence about user's "
            "preferences for Apple hardware, specifically Mac Studio "
            "and M2 Ultra for AI model running"
        )
    )

    assert items == ["Mac Studio", "M2 Ultra", "AI"]


def test_evidence_coverage_allows_or_target_alternatives() -> None:
    """Alternative-target slots should resolve when one option is covered."""

    coverage = assess_evidence_coverage(
        task=(
            "Conversation-evidence: retrieve messages mentioning "
            "'Apple', 'Mac Studio', or 'M2 Ultra'"
        ),
        evidence_items=[
            "The user considered a Mac Studio for local AI model work.",
        ],
        worker_resolved=True,
    )

    assert coverage["coverage_requirement"] == "any"
    assert coverage["evidence_quality"] == "confirmed"
    assert coverage["covered_items"] == ["Mac Studio"]
    assert "Apple" in coverage["missing_items"]


@pytest.mark.asyncio
async def test_memory_evidence_marks_partial_multi_target_result_unresolved() -> None:
    """Explicit multi-target memory slots need evidence for every target."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "memory_name": "model-alpha-price",
                    "content": "Model Alpha was stored at 1200.",
                    "source_kind": "manual",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "test"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            'Memory-evidence: retrieve durable evidence for '
            '"Model Alpha" and "Model Beta"'
        ),
        _base_context(),
    )

    assert result["resolved"] is False
    payload = result["result"]
    assert payload["coverage"]["evidence_quality"] == "partial"
    assert "Model Alpha" in payload["coverage"]["covered_items"]
    assert "Model Beta" in payload["coverage"]["missing_items"]
    assert payload["confirmed_evidence"] == []
    assert payload["partial_evidence"] == ["Model Alpha was stored at 1200."]


@pytest.mark.asyncio
async def test_live_context_current_local_time_requires_local_time_context() -> None:
    """Missing runtime local_time_context should stay unresolved without fallback."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("should not be called"))
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer active character current local time",
        _base_context(),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["local_time_context"]
    assert web_worker.calls == []
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_live_context_legacy_unknown_location_current_time_uses_runtime_state() -> None:
    """Legacy unknown-location current-time slots should normalize to runtime state."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("should not be called"))
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer current time for unknown location",
        _base_context(
            local_time_context={
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            }
        ),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_worker"] == "runtime_context_provider"
    assert result["result"]["missing_context"] == []
    assert web_worker.calls == []
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_live_context_current_user_local_time_without_user_time_context_is_unresolved() -> None:
    """User-local current time should require user_time_context, not location."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("should not be called"))
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer current user local time if configured",
        _base_context(
            local_time_context={
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            }
        ),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["user_time_context"]
    assert web_worker.calls == []
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_live_context_explicit_location_current_time_stays_external() -> None:
    """Explicit-place current time should remain on the external live path."""
    agent = LiveContextAgent()
    web_worker = _FakeWorker(_web_result("Auckland local time is 14:53 now."))
    memory_worker = _FakeWorker({"resolved": True, "result": []})
    conversation_worker = _FakeWorker({"resolved": True, "result": []})
    agent.web_agent = web_worker
    agent.memory_search_agent = memory_worker
    agent.conversation_search_agent = conversation_worker

    result = await agent.run(
        "Live-context: answer current time for explicit location Auckland",
        _base_context(
            local_time_context={
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            }
        ),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_worker"] == "web_agent3"
    assert len(web_worker.calls) == 1
    assert "fact_type=current_time" in web_worker.calls[0]["task"]
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_conversation_evidence_exact_phrase_uses_hybrid_search_and_refs() -> None:
    """Exact phrase evidence should use hybrid search and expose refs."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "约定就是约定, and here is https://example.test/post",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform_message_id": "msg-1",
                    "timestamp": "2026-05-01T23:00:00+00:00",
                    "role": "user",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = search_worker
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        'Conversation-evidence: find who said "约定就是约定"',
        _base_context(current_platform_message_id="msg-current"),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert keyword_worker.calls == []
    assert search_worker.calls[0]["context"]["exclude_current_question"] is True

    payload = result["result"]
    assert payload["primary_worker"] == "conversation_search_agent"
    assert payload["projection_payload"]["summaries"] == [
        "Tester at 2026-05-02 11:00: 约定就是约定, and here is https://example.test/post"
    ]
    assert payload["projection_payload"]["rows"][0]["summary"] == (
        "Tester at 2026-05-02 11:00: "
        "约定就是约定, and here is https://example.test/post"
    )
    assert {
        "ref_type": "person",
        "role": "speaker",
        "global_user_id": "user-1",
        "display_name": "Tester",
    } in payload["resolved_refs"]
    assert {
        "ref_type": "message",
        "platform_message_id": "msg-1",
        "timestamp": "2026-05-01T23:00:00+00:00",
        "global_user_id": "user-1",
        "display_name": "Tester",
    } in payload["resolved_refs"]
    assert {
        "ref_type": "url",
        "role": "posted_url",
        "url": "https://example.test/post",
    } in payload["resolved_refs"]


@pytest.mark.asyncio
async def test_conversation_evidence_cjk_quotes_use_hybrid_shortcut() -> None:
    """Quoted CJK phrases should route directly to hybrid search."""
    tasks = [
        "Conversation-evidence: find who said \u201c约定就是约定\u201d",
        "Conversation-evidence: find who said \u2018约定就是约定\u2019",
        "Conversation-evidence: find who said \u300c约定就是约定\u300d",
        "Conversation-evidence: find who said \u300e约定就是约定\u300f",
    ]

    for task in tasks:
        agent = ConversationEvidenceAgent()
        search_worker = _FakeWorker(
            {
                "resolved": True,
                "result": [
                    {
                        "body_text": "约定就是约定",
                        "display_name": "Tester",
                        "global_user_id": "user-1",
                        "platform_message_id": "msg-1",
                        "timestamp": "2026-05-01T23:00:00+00:00",
                        "role": "user",
                    }
                ],
                "attempts": 1,
                "cache": {"enabled": False, "hit": False, "reason": "open_range"},
            }
        )
        keyword_worker = _FakeWorker({"resolved": True, "result": []})
        agent.search_agent = search_worker
        agent.keyword_agent = keyword_worker

        result = await agent.run(
            task,
            _base_context(current_platform_message_id="msg-current"),
        )

        assert result["resolved"] is True
        assert result["result"]["primary_worker"] == "conversation_search_agent"
        assert len(search_worker.calls) == 1
        assert keyword_worker.calls == []


@pytest.mark.asyncio
async def test_conversation_evidence_semantic_topic_uses_search() -> None:
    """Fuzzy topic evidence belongs to the semantic conversation worker."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                (
                    0.73,
                    {
                        "body_text": "We talked about roller coaster plans.",
                        "display_name": "Tester",
                        "global_user_id": "user-1",
                        "platform_message_id": "msg-1",
                        "timestamp": "2026-05-01T22:00:00+00:00",
                    },
                )
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = search_worker
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        "Conversation-evidence: retrieve recent messages about roller coaster plans",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert keyword_worker.calls == []
    assert result["result"]["primary_worker"] == "conversation_search_agent"
    assert result["result"]["projection_payload"]["summaries"] == [
        "Tester at 2026-05-02 10:00: We talked about roller coaster plans."
    ]
    assert {
        "ref_type": "message",
        "platform_message_id": "msg-1",
        "timestamp": "2026-05-01T22:00:00+00:00",
        "global_user_id": "user-1",
        "display_name": "Tester",
    } in result["result"]["resolved_refs"]


@pytest.mark.asyncio
async def test_conversation_evidence_excludes_active_turn_keyword_row() -> None:
    """Current-turn keyword hits should not become conversation evidence."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "active question",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform": "qq",
                    "platform_channel_id": "chan-1",
                    "platform_message_id": "msg-current",
                    "timestamp": "2026-05-02T00:00:00+00:00",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    result = await agent.run(
        'Conversation-evidence: find who said "active question"',
        _base_context(active_turn_platform_message_ids=["msg-current"]),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["conversation_evidence"]
    assert result["result"]["selected_summary"] == ""
    assert result["result"]["projection_payload"]["summaries"] == []
    assert result["result"]["evidence"] == []
    assert result["result"]["resolved_refs"] == []


@pytest.mark.asyncio
async def test_conversation_evidence_excludes_active_turn_filter_row() -> None:
    """Current-turn recent-message hits should not become evidence."""
    agent = ConversationEvidenceAgent()
    filter_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "latest active message",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform": "qq",
                    "platform_channel_id": "chan-1",
                    "platform_message_id": "msg-current",
                    "timestamp": "2026-05-02T00:00:00+00:00",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.filter_agent = filter_worker

    result = await agent.run(
        "Conversation-evidence: retrieve recent messages speaker=current_user",
        _base_context(active_turn_platform_message_ids=["msg-current"]),
    )

    assert result["resolved"] is False
    assert result["result"]["primary_worker"] == "conversation_filter_agent"
    assert result["result"]["projection_payload"]["summaries"] == []
    assert result["result"]["resolved_refs"] == []


@pytest.mark.asyncio
async def test_conversation_evidence_excludes_active_turn_semantic_row() -> None:
    """Current-turn semantic hits should not become conversation evidence."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                (
                    0.91,
                    {
                        "body_text": "active topic question",
                        "display_name": "Tester",
                        "global_user_id": "user-1",
                        "platform": "qq",
                        "platform_channel_id": "chan-1",
                        "platform_message_id": "msg-current",
                        "timestamp": "2026-05-02T00:00:00+00:00",
                    },
                )
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        "Conversation-evidence: retrieve recent messages about active topic",
        _base_context(active_turn_platform_message_ids=["msg-current"]),
    )

    assert result["resolved"] is False
    assert result["result"]["primary_worker"] == "conversation_search_agent"
    assert result["result"]["projection_payload"]["summaries"] == []
    assert result["result"]["resolved_refs"] == []


@pytest.mark.asyncio
async def test_conversation_evidence_excludes_active_turn_row_id_hit() -> None:
    """Mongo row identity should filter active rows without platform IDs."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "conversation_row_id": "row-current",
                    "body_text": "active question",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform": "qq",
                    "platform_channel_id": "chan-1",
                    "platform_message_id": "",
                    "timestamp": "2026-05-02T00:00:00+00:00",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    result = await agent.run(
        'Conversation-evidence: find who said "active question"',
        _base_context(active_turn_conversation_row_ids=["row-current"]),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["conversation_evidence"]
    assert result["result"]["selected_summary"] == ""
    assert result["result"]["projection_payload"]["summaries"] == []
    assert result["result"]["evidence"] == []
    assert result["result"]["resolved_refs"] == []


@pytest.mark.asyncio
async def test_conversation_evidence_logs_active_turn_exclusion_reason_counts(
    caplog,
) -> None:
    """Active-turn exclusion telemetry should count reasons without ID values."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "conversation_row_id": "row-current",
                    "body_text": "active by row id",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform": "qq",
                    "platform_channel_id": "chan-1",
                    "platform_message_id": "",
                    "timestamp": "2026-05-02T00:00:00+00:00",
                },
                {
                    "body_text": "active by message id",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform": "qq",
                    "platform_channel_id": "chan-1",
                    "platform_message_id": "msg-current",
                    "timestamp": "2026-05-02T00:00:01+00:00",
                },
                {
                    "conversation_row_id": "row-old",
                    "body_text": "historical evidence",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform": "qq",
                    "platform_channel_id": "chan-1",
                    "platform_message_id": "msg-old",
                    "timestamp": "2026-05-01T23:59:00+00:00",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    with caplog.at_level(
        logging.INFO,
        logger="kazusa_ai_chatbot.rag.conversation_evidence.agent",
    ):
        result = await agent.run(
            'Conversation-evidence: find who said "historical evidence"',
            _base_context(
                active_turn_conversation_row_ids=["row-current"],
                active_turn_platform_message_ids=["msg-current"],
            ),
        )

    assert result["resolved"] is True
    assert result["result"]["projection_payload"]["summaries"] == [
        "Tester at 2026-05-02 11:59: historical evidence"
    ]
    assert "excluded_active_turn_rows=2" in caplog.text
    assert "excluded_by_conversation_row_id=1" in caplog.text
    assert "excluded_by_platform_message_id=1" in caplog.text
    assert "row-current" not in caplog.text
    assert "msg-current" not in caplog.text


@pytest.mark.asyncio
async def test_conversation_evidence_excludes_collapsed_active_turn_rows() -> None:
    """Collapsed active-turn source messages should all be removed."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "same body",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform_message_id": "msg-1",
                    "timestamp": "2026-05-02T00:00:00+00:00",
                },
                {
                    "body_text": "same body",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform_message_id": "msg-2",
                    "timestamp": "2026-05-02T00:00:01+00:00",
                },
                {
                    "body_text": "same body",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform_message_id": "msg-old",
                    "timestamp": "2026-05-01T23:59:00+00:00",
                },
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    result = await agent.run(
        'Conversation-evidence: find who said "same body"',
        _base_context(active_turn_platform_message_ids=["msg-1", "msg-2"]),
    )

    assert result["resolved"] is True
    assert result["result"]["projection_payload"]["summaries"] == [
        "Tester at 2026-05-02 11:59: same body"
    ]
    assert {
        "ref_type": "message",
        "platform_message_id": "msg-old",
        "timestamp": "2026-05-01T23:59:00+00:00",
        "global_user_id": "user-1",
        "display_name": "Tester",
    } in result["result"]["resolved_refs"]


@pytest.mark.asyncio
async def test_conversation_evidence_keeps_row_without_message_id() -> None:
    """Rows without platform message IDs should not be fallback-filtered."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "active question",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "timestamp": "2026-05-02T00:00:00+00:00",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    result = await agent.run(
        'Conversation-evidence: find who said "active question"',
        _base_context(active_turn_platform_message_ids=["msg-current"]),
    )

    assert result["resolved"] is True
    assert result["result"]["projection_payload"]["summaries"] == [
        "Tester at 2026-05-02 12:00: active question"
    ]
    assert {
        "ref_type": "message",
        "platform_message_id": "",
        "timestamp": "2026-05-02T00:00:00+00:00",
        "global_user_id": "user-1",
        "display_name": "Tester",
    } in result["result"]["resolved_refs"]


@pytest.mark.asyncio
async def test_conversation_evidence_does_not_match_empty_active_row_id() -> None:
    """Empty row IDs are absence markers, not comparable active-turn IDs."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "conversation_row_id": "",
                    "body_text": "blank row id should stay",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                    "platform": "qq",
                    "platform_channel_id": "chan-1",
                    "platform_message_id": "",
                    "timestamp": "2026-05-02T00:00:00+00:00",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    result = await agent.run(
        'Conversation-evidence: find who said "blank row id should stay"',
        _base_context(
            active_turn_conversation_row_ids=[""],
            active_turn_platform_message_ids=[""],
        ),
    )

    assert result["resolved"] is True
    assert result["result"]["projection_payload"]["summaries"] == [
        "Tester at 2026-05-02 12:00: blank row id should stay"
    ]


@pytest.mark.asyncio
async def test_conversation_evidence_filter_uses_resolved_person_ref() -> None:
    """Known person refs should be passed as structured worker context."""
    agent = ConversationEvidenceAgent()
    filter_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "I posted a link yesterday.",
                    "display_name": "Resolved User",
                    "global_user_id": "resolved-user",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.filter_agent = filter_worker

    context = _base_context(
        known_facts=[
            {
                "slot": "Person-context: resolve display name",
                "raw_result": {
                    "resolved_refs": [
                        {
                            "ref_type": "person",
                            "role": "profile_owner",
                            "global_user_id": "resolved-user",
                            "display_name": "Resolved User",
                        }
                    ]
                },
            }
        ]
    )

    result = await agent.run(
        "Conversation-evidence: retrieve recent messages from the user resolved in slot 1",
        context,
    )

    assert result["resolved"] is True
    assert len(filter_worker.calls) == 1
    assert filter_worker.calls[0]["context"]["global_user_id"] == "resolved-user"
    assert result["result"]["primary_worker"] == "conversation_filter_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_current_user_scope_ignores_selector_person_ref(
    monkeypatch,
) -> None:
    """Current-user scope should not let the selector invent a person dependency."""

    class _FakeSelectorLLM:
        """Selector test double that returns the observed bad dependency flag."""

        async def ainvoke(self, _messages: list) -> SimpleNamespace:
            """Return a valid selector payload with a bad person-ref requirement."""
            return_value = SimpleNamespace(
                content=(
                    '{"worker": "conversation_search_agent", '
                    '"reason": "semantic message evidence", '
                    '"requires_person_ref": true}'
                )
            )
            return return_value

    monkeypatch.setattr(
        conversation_evidence_module,
        "_selector_llm",
        _FakeSelectorLLM(),
    )
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "The user described explicit accusations.",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve messages from current_user "
            "containing sexual harassment or explicit accusations "
            "speaker=current_user"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert search_worker.calls[0]["context"]["global_user_id"] == "user-1"
    assert result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_current_user_scope_replaces_self_dependency_keyword() -> None:
    """Current-user content search should not require a fake prior person slot."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "The user mentioned dessert.",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    old_result = await agent.run(
        (
            "Conversation-evidence: retrieve recent messages from the user "
            "resolved in slot 1 containing exact term 'dessert'"
        ),
        _base_context(),
    )
    context_with_prior_person = _base_context(
        known_facts=[
            {
                "slot": "Person-context: resolve display name",
                "raw_result": {
                    "resolved_refs": [
                        {
                            "ref_type": "person",
                            "global_user_id": "other-user",
                            "display_name": "Other",
                        }
                    ]
                },
            }
        ]
    )
    new_result = await agent.run(
        (
            "Conversation-evidence: retrieve recent messages containing "
            "exact term 'dessert' speaker=current_user"
        ),
        context_with_prior_person,
    )

    assert old_result["resolved"] is False
    assert old_result["result"]["missing_context"] == ["person_ref"]
    assert new_result["resolved"] is True
    assert len(keyword_worker.calls) == 1
    assert keyword_worker.calls[0]["context"]["global_user_id"] == "user-1"
    assert new_result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_current_user_scope_replaces_self_dependency_url() -> None:
    """Current-user URL search should bind runtime user context directly."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "I shared https://example.test earlier.",
                    "display_name": "Tester",
                    "global_user_id": "user-1",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    old_result = await agent.run(
        (
            "Conversation-evidence: retrieve messages from the user resolved "
            "in slot 1 containing a URL"
        ),
        _base_context(),
    )
    new_result = await agent.run(
        "Conversation-evidence: retrieve messages containing a URL speaker=current_user",
        _base_context(),
    )

    assert old_result["resolved"] is False
    assert old_result["result"]["missing_context"] == ["person_ref"]
    assert new_result["resolved"] is True
    assert len(keyword_worker.calls) == 1
    assert keyword_worker.calls[0]["context"]["global_user_id"] == "user-1"
    assert new_result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_current_user_scope_replaces_self_dependency_topic() -> None:
    """Current-user topic search should not be encoded as a prior-slot dependency."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                (
                    0.71,
                    {
                        "body_text": "I talked about train plans.",
                        "display_name": "Tester",
                        "global_user_id": "user-1",
                    },
                )
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = search_worker

    old_result = await agent.run(
        (
            "Conversation-evidence: retrieve recent messages from the user "
            "resolved in slot 1 about train plans"
        ),
        _base_context(),
    )
    new_result = await agent.run(
        (
            "Conversation-evidence: retrieve recent messages about train "
            "plans speaker=current_user"
        ),
        _base_context(),
    )

    assert old_result["resolved"] is False
    assert old_result["result"]["missing_context"] == ["person_ref"]
    assert new_result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert search_worker.calls[0]["context"]["global_user_id"] == "user-1"
    assert new_result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_any_speaker_scope_removes_current_user_filter() -> None:
    """Any-speaker searches should not inherit the current user as a filter."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                (
                    0.69,
                    {
                        "body_text": "Someone talked about train plans.",
                        "display_name": "Other",
                        "global_user_id": "other-user",
                    },
                )
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve recent messages about train "
            "plans speaker=any_speaker"
        ),
        _base_context(display_name="Tester"),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert "global_user_id" not in search_worker.calls[0]["context"]
    assert "display_name" not in search_worker.calls[0]["context"]
    assert result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_unscoped_search_removes_current_user_filter() -> None:
    """Unscoped evidence searches should default to the group, not the caller."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                (
                    0.69,
                    {
                        "body_text": "Someone talked about train plans.",
                        "display_name": "Other",
                        "global_user_id": "other-user",
                    },
                )
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        "Conversation-evidence: retrieve recent messages about train plans",
        _base_context(display_name="Tester"),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert "global_user_id" not in search_worker.calls[0]["context"]
    assert "display_name" not in search_worker.calls[0]["context"]
    assert "conversation_user_scope" not in search_worker.calls[0]["context"]
    assert result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_active_character_scope_uses_character_identity() -> None:
    """Active-character searches should filter worker calls to character rows."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "I said the project might slip.",
                    "display_name": "Kazusa",
                    "global_user_id": "character-1",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve messages containing exact term "
            "'project' speaker=active_character"
        ),
        _base_context(
            character_profile={
                "global_user_id": "character-1",
                "name": "Kazusa",
            },
        ),
    )

    assert result["resolved"] is True
    assert len(keyword_worker.calls) == 1
    worker_context = keyword_worker.calls[0]["context"]
    assert worker_context["global_user_id"] == "character-1"
    assert worker_context["display_name"] == "Kazusa"
    assert result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_current_episode_active_character_definition_request_uses_conversation_worker() -> None:
    """Current-episode transcript proof is chat evidence, not Recall state."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": '你先解释一下呗，我等你的定义呢',
                    "display_name": "Kazusa",
                    "global_user_id": "character-1",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = search_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve recent messages from the "
            "current episode where the active character asked about yandere "
            "definition speaker=active_character"
        ),
        _base_context(
            character_profile={
                "global_user_id": "character-1",
                "name": "Kazusa",
            },
        ),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    worker_context = search_worker.calls[0]["context"]
    assert worker_context["global_user_id"] == "character-1"
    assert worker_context["display_name"] == "Kazusa"
    assert result["result"]["primary_worker"] == "conversation_search_agent"


@pytest.mark.asyncio
async def test_conversation_evidence_active_character_scope_warns_without_identity(
    caplog,
) -> None:
    """Missing active-character identity should be visible but non-fatal."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "body_text": "I said the project might slip.",
                    "display_name": "Kazusa",
                    "global_user_id": "character-1",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.search_agent = keyword_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve messages containing exact term "
            "'project' speaker=active_character"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(keyword_worker.calls) == 1
    worker_context = keyword_worker.calls[0]["context"]
    assert "global_user_id" not in worker_context
    assert "display_name" not in worker_context
    assert (
        "conversation_evidence: speaker=active_character requested "
        "without character_profile"
    ) in caplog.text
    assert "platform_channel_id=chan-1" in caplog.text


@pytest.mark.asyncio
async def test_conversation_evidence_count_uses_aggregate() -> None:
    """Count and ranking requests should go to the aggregate worker."""
    agent = ConversationEvidenceAgent()
    aggregate_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "aggregate": "count_by_user",
                "time_window": "recent",
                "total_count": 5,
                "rows": [
                    {
                        "global_user_id": "user-1",
                        "platform_user_id": "673225019",
                        "display_names": ["Tester"],
                        "message_count": 5,
                        "last_timestamp": "2026-05-01T22:00:00+00:00",
                    }
                ],
            },
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    agent.aggregate_agent = aggregate_worker

    result = await agent.run(
        "Conversation-evidence: count recent messages mentioning cookie manager by user",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(aggregate_worker.calls) == 1
    assert result["result"]["primary_worker"] == "conversation_aggregate_agent"
    assert result["result"]["projection_payload"]["summaries"] == [
        "count_by_user, window=recent, total=5, top rows: "
        "Tester, 5 messages, last=2026-05-02 10:00"
    ]
    assert {
        "ref_type": "person",
        "role": "aggregate_subject",
        "global_user_id": "user-1",
        "display_name": "Tester",
    } in result["result"]["resolved_refs"]


@pytest.mark.asyncio
async def test_conversation_evidence_account_limit_discussion_uses_search() -> None:
    """Account-limit topic text should not be mistaken for count aggregation."""
    agent = ConversationEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "display_name": "Tester",
                    "body_text": "5小时限制才5%周限",
                    "timestamp": "2026-05-22T21:07:15+00:00",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": False, "hit": False, "reason": "open_range"},
        }
    )
    aggregate_worker = _FakeWorker({"resolved": True, "result": {}})
    agent.search_agent = search_worker
    agent.aggregate_agent = aggregate_worker

    result = await agent.run(
        (
            "Conversation-evidence: retrieve subsequent discussion about "
            "free account weekly limits, 5-hour limits, and 5% weekly limits"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert aggregate_worker.calls == []
    assert result["result"]["primary_worker"] == "conversation_search_agent"
    assert "5小时限制才5%周限" in result["result"]["selected_summary"]


@pytest.mark.asyncio
async def test_conversation_evidence_rejects_active_agreement_intent() -> None:
    """Active episode agreement lookup belongs to Recall, not chat search."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = keyword_worker

    result = await agent.run(
        "Conversation-evidence: retrieve active agreement for today's appointment",
        _base_context(),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["incompatible_intent:Recall"]
    assert keyword_worker.calls == []


@pytest.mark.asyncio
async def test_conversation_evidence_rejects_episode_state_intent() -> None:
    """Current episode state lookup belongs to Recall, not chat search."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = keyword_worker

    result = await agent.run(
        "Conversation-evidence: retrieve where the current episode left off",
        _base_context(),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["incompatible_intent:Recall"]
    assert keyword_worker.calls == []


@pytest.mark.asyncio
async def test_memory_evidence_official_address_uses_search() -> None:
    """Natural-language address facts should use semantic memory evidence."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "memory_name": "active-character-official-address",
                    "content": "The active character's official address is 123 Example Street.",
                    "source_kind": "seeded_manual",
                    "memory_type": "fact",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = keyword_worker
    agent.search_agent = search_worker

    result = await agent.run(
        "Memory-evidence: retrieve durable evidence about the active character's official address",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert keyword_worker.calls == []
    assert result["result"]["primary_worker"] == "persistent_memory_search_agent"
    assert result["result"]["projection_payload"]["memory_rows"] == [
        {
            "memory_name": "active-character-official-address",
            "content": "The active character's official address is 123 Example Street.",
            "source_kind": "seeded_manual",
            "memory_type": "fact",
        }
    ]
    assert result["result"]["resolved_refs"] == [
        {
            "ref_type": "memory",
            "memory_name": "active-character-official-address",
            "source_kind": "seeded_manual",
        }
    ]


@pytest.mark.asyncio
async def test_memory_evidence_exact_memory_name_uses_hybrid_search() -> None:
    """Literal memory identifiers should use shared hybrid memory evidence."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "memory_name": "active-character-official-address",
                    "content": "The active character's official address is 123 Example Street.",
                    "source_kind": "seeded_manual",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = search_worker
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        "Memory-evidence: exact memory_name active-character-official-address",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert keyword_worker.calls == []
    assert result["result"]["primary_worker"] == "persistent_memory_search_agent"


@pytest.mark.asyncio
async def test_memory_evidence_common_sense_uses_search() -> None:
    """Fuzzy common-sense memory belongs to semantic memory search."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "memory_name": "short-walk-common-sense",
                    "content": "A 50 meter trip is usually short enough to walk.",
                    "source_kind": "seeded_manual",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = search_worker
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        "Memory-evidence: retrieve common-sense evidence relevant to choosing walk vs drive for a 50 meter trip",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert keyword_worker.calls == []
    assert result["result"]["primary_worker"] == "persistent_memory_search_agent"


@pytest.mark.asyncio
async def test_memory_evidence_unresolved_candidates_are_observation_only() -> None:
    """Unresolved durable-memory candidates are not accepted evidence."""
    agent = MemoryEvidenceAgent()
    stale_policy_row = {
        "memory_name": "volatile-technical-data-policy",
        "content": (
            "Latest device, model, driver, price, and benchmark comparisons "
            "should use fresh retrieval because memory can be stale."
        ),
        "source_kind": "seeded_manual",
    }
    search_worker = _FakeWorker(
        {
            "resolved": False,
            "result": [stale_policy_row],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = search_worker
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        "Memory-evidence: retrieve durable evidence about device performance comparison",
        _base_context(),
    )

    payload = result["result"]
    assert result["resolved"] is False
    assert payload["selected_summary"] == ""
    assert payload["evidence"] == []
    assert payload["projection_payload"]["memory_rows"] == []
    assert payload["observation_candidates"] == [
        {
            "content": stale_policy_row["content"],
            "source": "memory:memory_name:volatile-technical-data-policy",
        }
    ]
    assert payload["source_hints"] == [
        {
            "kind": "memory",
            "source": "memory:memory_name:volatile-technical-data-policy",
        }
    ]
    assert payload["missing_context"] == ["memory_evidence"]


@pytest.mark.asyncio
async def test_memory_evidence_resolved_rows_remain_accepted_evidence() -> None:
    """Resolved durable-memory rows keep their evidence payload contract."""
    agent = MemoryEvidenceAgent()
    accepted_row = {
        "memory_name": "official-address",
        "content": "The active character's official address is 123 Example Street.",
        "source_kind": "seeded_manual",
    }
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [accepted_row],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = search_worker
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        "Memory-evidence: retrieve durable evidence about the official address",
        _base_context(),
    )

    payload = result["result"]
    assert result["resolved"] is True
    assert payload["selected_summary"] == accepted_row["content"]
    assert payload["evidence"] == [accepted_row["content"]]
    assert payload["projection_payload"]["memory_rows"] == [accepted_row]
    assert payload["observation_candidates"] == []
    assert payload["missing_context"] == []


def test_memory_evidence_selector_prompt_renders_scoped_worker_option() -> None:
    """The selector prompt should expose the scoped user-memory worker."""
    prompt = memory_evidence_module._SELECTOR_PROMPT

    assert "# 输入格式" in prompt
    assert "# 输出格式" in prompt
    assert "user_memory_evidence_agent" in prompt
    assert "当前用户 durable memory" in prompt


@pytest.mark.asyncio
async def test_memory_evidence_user_memory_unit_uses_scoped_worker() -> None:
    """Current-user continuity should route to the scoped user-memory worker."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The current user prefers jasmine tea.",
                "memory_rows": [
                    {
                        "unit_id": "unit-1",
                        "unit_type": "objective_fact",
                        "fact": "The current user prefers jasmine tea.",
                        "subjective_appraisal": "Kazusa sees this as stable continuity.",
                        "relationship_signal": "Keep tea continuity available.",
                        "content": "The current user prefers jasmine tea.",
                        "updated_at": "2026-05-03T00:00:00+00:00",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                        "authority": "scoped_continuity",
                        "truth_status": "character_lore_or_interaction_continuity",
                        "origin": "consolidated_interaction",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker

    result = await agent.run(
        "Memory-evidence: retrieve durable user memory evidence about the current user's accepted preference",
        _base_context(),
    )

    assert result["resolved"] is True
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"
    assert result["result"]["projection_payload"]["memory_rows"][0]["content"] == (
        "The current user prefers jasmine tea."
    )
    assert result["result"]["projection_payload"]["memory_rows"][0]["scope_type"] == (
        "user_continuity"
    )


@pytest.mark.asyncio
async def test_memory_evidence_current_user_preferences_use_scoped_worker() -> None:
    """Current-user durable preferences should route to scoped user memory."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The current user prioritizes search precision.",
                "memory_rows": [
                    {
                        "unit_id": "unit-preference",
                        "unit_type": "objective_fact",
                        "fact": "The current user prioritizes search precision.",
                        "content": "The current user prioritizes search precision.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about current_user's "
            "technical preferences regarding RAG/GraphRAG/information graph "
            "solutions"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"
    assert result["result"]["projection_payload"]["memory_rows"][0]["unit_id"] == (
        "unit-preference"
    )


@pytest.mark.asyncio
async def test_memory_evidence_commitment_status_uses_scoped_worker() -> None:
    """Current-user commitment lifecycle status should search scoped memory."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": (
                    "The current user fulfilled a dessert-fare commitment."
                ),
                "memory_rows": [
                    {
                        "unit_id": "unit-dessert-fare",
                        "unit_type": "milestone",
                        "fact": (
                            "The current user fulfilled a dessert-fare "
                            "commitment."
                        ),
                        "content": (
                            "The current user fulfilled a dessert-fare "
                            "commitment."
                        ),
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                        "status": "completed",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about any completed "
            "or outstanding dessert promises/commitments involving the "
            "current user"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"
    assert result["result"]["projection_payload"]["memory_rows"][0]["status"] == (
        "completed"
    )


@pytest.mark.asyncio
async def test_memory_evidence_selector_can_route_user_preference_slot(
    monkeypatch,
) -> None:
    """Ambiguous user-preference slots should reach the selector LLM."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The current user prioritizes recall precision.",
                "memory_rows": [
                    {
                        "unit_id": "unit-preference",
                        "content": "The current user prioritizes recall precision.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    selector_calls: list[list[object]] = []

    class _SelectorLLM:
        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            selector_calls.append(messages)
            response = SimpleNamespace(
                content=(
                    '{"worker": "user_memory_evidence_agent", '
                    '"reason": "scoped current-user continuity evidence"}'
                )
            )
            return response

    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker
    monkeypatch.setattr(memory_evidence_module, "_selector_llm", _SelectorLLM())

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about user's "
            "technical preferences regarding RAG/GraphRAG/information graph "
            "solutions"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(selector_calls) == 1
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"


@pytest.mark.asyncio
async def test_memory_evidence_selector_can_route_user_promise_slot(
    monkeypatch,
) -> None:
    """Ambiguous user-promise slots should reach source selection."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The user's food promise is scoped continuity.",
                "memory_rows": [
                    {
                        "unit_id": "unit-promise",
                        "content": "The user's food promise is scoped continuity.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    selector_calls: list[list[object]] = []

    class _SelectorLLM:
        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            selector_calls.append(messages)
            response = SimpleNamespace(
                content=(
                    '{"worker": "user_memory_evidence_agent", '
                    '"reason": "scoped current-user continuity evidence"}'
                )
            )
            return response

    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker
    monkeypatch.setattr(memory_evidence_module, "_selector_llm", _SelectorLLM())

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about the user's "
            "past promises or commitments regarding food/cooking"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(selector_calls) == 1
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"


@pytest.mark.asyncio
async def test_memory_evidence_selector_can_route_user_consideration_slot(
    monkeypatch,
) -> None:
    """Ambiguous user-consideration slots should reach source selection."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The user's purchase consideration is scoped.",
                "memory_rows": [
                    {
                        "unit_id": "unit-consideration",
                        "content": "The user's purchase consideration is scoped.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    selector_calls: list[list[object]] = []

    class _SelectorLLM:
        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            selector_calls.append(messages)
            response = SimpleNamespace(
                content=(
                    '{"worker": "user_memory_evidence_agent", '
                    '"reason": "scoped current-user continuity evidence"}'
                )
            )
            return response

    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker
    monkeypatch.setattr(memory_evidence_module, "_selector_llm", _SelectorLLM())

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about user's previous "
            "considerations regarding Apple hardware for AI models"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(selector_calls) == 1
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"


@pytest.mark.asyncio
async def test_memory_evidence_unclear_slot_with_query_uses_selector(
    monkeypatch,
) -> None:
    """Unclear memory slots should not default to shared memory in context."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The user considered a Mac Studio.",
                "memory_rows": [
                    {
                        "unit_id": "unit-hardware",
                        "content": "The user considered a Mac Studio.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    selector_calls: list[list[object]] = []

    class _SelectorLLM:
        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            selector_calls.append(messages)
            response = SimpleNamespace(
                content=(
                    '{"worker": "user_memory_evidence_agent", '
                    '"reason": "scoped current-user continuity evidence"}'
                )
            )
            return response

    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker
    monkeypatch.setattr(memory_evidence_module, "_selector_llm", _SelectorLLM())

    result = await agent.run(
        "Memory-evidence: retrieve durable evidence about Apple machines for running local AI models",
        _base_context(
            original_query=(
                "The user asks which Apple machine they previously "
                "considered for local AI model work."
            ),
        ),
    )

    assert result["resolved"] is True
    assert len(selector_calls) == 1
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"


@pytest.mark.asyncio
async def test_memory_evidence_remember_me_slot_uses_scoped_worker() -> None:
    """Current-user recognition requests should search scoped user memory."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The current user and the active character have prior shared interactions.",
                "memory_rows": [
                    {
                        "unit_id": "unit-remember-me",
                        "unit_type": "objective_fact",
                        "fact": "The current user and the active character have prior shared interactions.",
                        "content": "The current user and the active character have prior shared interactions.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about the current "
            "user's identity and past interactions with active character"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"
    assert result["result"]["projection_payload"]["memory_rows"][0]["unit_id"] == (
        "unit-remember-me"
    )


@pytest.mark.asyncio
async def test_memory_evidence_current_user_shared_media_uses_scoped_worker() -> None:
    """Media shared by the current user with the character is private continuity."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "The current user shared a remembered image.",
                "memory_rows": [
                    {
                        "unit_id": "unit-shared-media",
                        "unit_type": "objective_fact",
                        "fact": "The current user shared a remembered image.",
                        "content": "The current user shared a remembered image.",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about illustrations "
            "or photos shared by current user with active character"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"
    assert result["result"]["projection_payload"]["memory_rows"][0]["unit_id"] == (
        "unit-shared-media"
    )


@pytest.mark.asyncio
async def test_memory_evidence_official_character_fact_stays_shared_memory() -> None:
    """Official character facts should remain shared durable memory lookups."""
    agent = MemoryEvidenceAgent()
    accepted_row = {
        "memory_name": "active-character-official-address",
        "content": "The active character's official address is 123 Example Street.",
        "source_kind": "seeded_manual",
    }
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [accepted_row],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    user_worker = _FakeWorker({"resolved": True, "result": {"memory_rows": []}})
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker

    result = await agent.run(
        (
            "Memory-evidence: retrieve durable evidence about the active "
            "character's official address"
        ),
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert user_worker.calls == []
    assert result["result"]["primary_worker"] == "persistent_memory_search_agent"
    assert result["result"]["projection_payload"]["memory_rows"] == [accepted_row]


@pytest.mark.asyncio
async def test_memory_evidence_shared_fact_ignores_remember_me_query_scope() -> None:
    """A separate shared-memory slot must not inherit remember-me scope."""
    agent = MemoryEvidenceAgent()
    accepted_row = {
        "memory_name": "active-character-official-address",
        "content": "The active character's official address is 123 Example Street.",
        "source_kind": "seeded_manual",
    }
    search_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [accepted_row],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    user_worker = _FakeWorker({"resolved": True, "result": {"memory_rows": []}})
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker
    task = (
        "Memory-evidence: retrieve durable evidence about the active "
        "character's official address"
    )

    result = await agent.run(
        task,
        _base_context(
            original_query='<character mention> 你还记得我吗？你家的官方地址是什么？',
            current_slot=task,
        ),
    )

    assert result["resolved"] is True
    assert len(search_worker.calls) == 1
    assert user_worker.calls == []
    assert result["result"]["primary_worker"] == "persistent_memory_search_agent"
    assert result["result"]["projection_payload"]["memory_rows"] == [accepted_row]


@pytest.mark.asyncio
async def test_memory_evidence_old_setting_slot_uses_selector_context(
    monkeypatch,
) -> None:
    """Old durable-setting slots should use original-query scoped continuity."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    user_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "selected_summary": "冰淇淋摊老板是千纱的初中学姐。",
                "memory_rows": [
                    {
                        "unit_id": "unit-xuejie",
                        "unit_type": "objective_fact",
                        "fact": "冰淇淋摊老板是千纱的初中学姐。",
                        "subjective_appraisal": "Kazusa treats this as scoped continuity.",
                        "relationship_signal": "Keep this lore scoped to this user.",
                        "content": "冰淇淋摊老板是千纱的初中学姐。",
                        "updated_at": "2026-05-03T00:00:00+00:00",
                        "source_system": "user_memory_units",
                        "scope_type": "user_continuity",
                        "scope_global_user_id": "user-1",
                        "authority": "scoped_continuity",
                        "truth_status": "character_lore_or_interaction_continuity",
                        "origin": "consolidated_interaction",
                    }
                ],
                "source_system": "user_memory_units",
                "scope_type": "user_continuity",
                "scope_global_user_id": "user-1",
                "missing_context": [],
            },
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "reason": "scoped_user_memory_uncached",
            },
        }
    )
    agent.search_agent = search_worker
    agent.user_memory_agent = user_worker
    selector_calls: list[list[object]] = []

    class _SelectorLLM:
        async def ainvoke(self, messages: list[object]) -> SimpleNamespace:
            selector_calls.append(messages)
            response = SimpleNamespace(
                content=(
                    '{"worker": "user_memory_evidence_agent", '
                    '"reason": "scoped current-user continuity evidence"}'
                )
            )
            return response

    monkeypatch.setattr(memory_evidence_module, "_selector_llm", _SelectorLLM())

    result = await agent.run(
        'Memory-evidence: retrieve durable evidence about “学姐抹茶冰淇淋店” setting',
        _base_context(
            original_query='请根据你和当前用户之间已经形成的私有连续性，回忆一下“学姐抹茶冰淇淋店”那个设定。',
        ),
    )

    assert result["resolved"] is True
    assert len(selector_calls) == 1
    assert search_worker.calls == []
    assert len(user_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_memory_evidence_agent"


@pytest.mark.asyncio
async def test_memory_evidence_rejects_live_external_fact() -> None:
    """Live facts must stay outside durable memory evidence."""
    agent = MemoryEvidenceAgent()
    search_worker = _FakeWorker({"resolved": True, "result": []})
    agent.search_agent = search_worker

    result = await agent.run(
        "Memory-evidence: retrieve current weather for the active character's location",
        _base_context(),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["incompatible_intent:Live-context"]
    assert search_worker.calls == []


@pytest.mark.asyncio
async def test_memory_evidence_rejects_active_agreement() -> None:
    """Active promises and agreements belong to Recall."""
    agent = MemoryEvidenceAgent()
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        "Memory-evidence: retrieve active agreement about today's appointment",
        _base_context(),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == ["incompatible_intent:Recall"]
    assert keyword_worker.calls == []


@pytest.mark.asyncio
async def test_person_context_identity_lookup_emits_person_ref() -> None:
    """Display-name identity lookup should expose a structured person ref."""
    agent = PersonContextAgent()
    lookup_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "global_user_id": "person-1",
                "display_name": "Named User",
                "platform": "qq",
            },
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    agent.lookup_agent = lookup_worker

    result = await agent.run(
        "Person-context: resolve display name Named User",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(lookup_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_lookup_agent"
    assert result["result"]["resolved_refs"] == [
        {
            "ref_type": "person",
            "role": "profile_owner",
            "global_user_id": "person-1",
            "display_name": "Named User",
        }
    ]


@pytest.mark.asyncio
async def test_person_context_display_name_profile_chain_is_authorized() -> None:
    """Display-name profile requests may use lookup then profile."""
    agent = PersonContextAgent()
    lookup_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "global_user_id": "person-2",
                "display_name": '小钳子',
            },
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    profile_payload = {
        "global_user_id": "person-2",
        "display_name": '小钳子',
        "self_image": {"summary": "quiet but curious"},
        "_user_memory_units": [{"fact": "likes tea"}],
    }
    profile_worker = _FakeWorker(
        {
            "resolved": True,
            "result": profile_payload,
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    agent.lookup_agent = lookup_worker
    agent.profile_agent = profile_worker

    result = await agent.run(
        'Person-context: retrieve profile/impression for display name 小钳子',
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(lookup_worker.calls) == 1
    assert len(profile_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_profile_agent"
    assert result["result"]["supporting_workers"] == ["user_lookup_agent"]
    projection_payload = result["result"]["projection_payload"]
    assert projection_payload["profile_kind"] == "third_party"
    assert projection_payload["owner_global_user_id"] == "person-2"
    assert projection_payload["profile"] == profile_payload


@pytest.mark.asyncio
async def test_person_context_current_user_profile_uses_profile_worker() -> None:
    """Current-user profile requests should preserve the profile payload."""
    agent = PersonContextAgent()
    profile_payload = {
        "global_user_id": "user-1",
        "display_name": "Tester",
        "self_image": {"summary": "current user image"},
    }
    profile_worker = _FakeWorker(
        {
            "resolved": True,
            "result": profile_payload,
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    agent.profile_agent = profile_worker

    result = await agent.run(
        "Person-context: retrieve current user profile",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(profile_worker.calls) == 1
    assert result["result"]["projection_payload"]["profile_kind"] == "current_user"
    assert result["result"]["projection_payload"]["profile"] == profile_payload


@pytest.mark.asyncio
async def test_person_context_user_list_projects_summary() -> None:
    """Display-name predicate enumeration belongs to user-list worker."""
    agent = PersonContextAgent()
    list_worker = _FakeWorker(
        {
            "resolved": True,
            "result": {
                "users": [
                    {"global_user_id": "u-1", "display_name": "Alice"},
                    {"global_user_id": "u-2", "display_name": "Annie"},
                ],
                "summary": "Alice and Annie match the display-name predicate.",
            },
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    agent.user_list_agent = list_worker

    result = await agent.run(
        "Person-context: list users whose display names start with A",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(list_worker.calls) == 1
    assert result["result"]["primary_worker"] == "user_list_agent"
    assert result["result"]["projection_payload"]["profile_kind"] == "user_list"
    assert "Alice and Annie" in result["result"]["projection_payload"]["summary"]


@pytest.mark.asyncio
async def test_person_context_relationship_projects_summary() -> None:
    """Relationship rankings should use the relationship worker."""
    agent = PersonContextAgent()
    relationship_worker = _FakeWorker(
        {
            "resolved": True,
            "result": [
                {
                    "global_user_id": "user-1",
                    "display_name": "Tester",
                    "relationship_label": "Unwavering",
                }
            ],
            "attempts": 1,
            "cache": {"enabled": True, "hit": False, "reason": "miss_stored"},
        }
    )
    agent.relationship_agent = relationship_worker

    result = await agent.run(
        "Person-context: rank users by active character relationship from top limit 1",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(relationship_worker.calls) == 1
    assert result["result"]["projection_payload"]["profile_kind"] == "relationship"
    assert "Unwavering" in result["result"]["projection_payload"]["summary"]


@pytest.mark.asyncio
async def test_person_context_rejects_unknown_speaker_message_search() -> None:
    """Unknown speaker discovery by phrase belongs to Conversation-evidence."""
    agent = PersonContextAgent()
    lookup_worker = _FakeWorker({"resolved": True, "result": {}})
    agent.lookup_agent = lookup_worker

    result = await agent.run(
        'Person-context: find unknown speaker who said "约定就是约定"',
        _base_context(),
    )

    assert result["resolved"] is False
    assert result["result"]["missing_context"] == [
        "incompatible_intent:Conversation-evidence"
    ]
    assert lookup_worker.calls == []


@pytest.mark.asyncio
async def test_top_level_capability_logs_info_and_debug(caplog) -> None:
    """Capability logs should keep summary at INFO and raw payloads at DEBUG."""
    agent = LiveContextAgent()
    agent.web_agent = _FakeWorker(_web_result("Auckland is 17 C now."))

    with caplog.at_level("DEBUG", logger="kazusa_ai_chatbot.rag.live_context.agent"):
        await agent.run(
            "Live-context: answer current temperature for explicit location Auckland",
            _base_context(),
        )

    info_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "INFO"
    ]
    debug_messages = [
        record.getMessage()
        for record in caplog.records
        if record.levelname == "DEBUG"
    ]

    assert any("live_context_agent output" in message for message in info_messages)
    assert any("selected_summary=Auckland is 17 C now." in message for message in info_messages)
    assert not any("worker_payloads" in message for message in info_messages)
    assert any("worker_payloads" in message for message in debug_messages)
