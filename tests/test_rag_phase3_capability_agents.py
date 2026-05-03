"""Deterministic contract tests for RAG2 top-level capability agents."""

from __future__ import annotations

import pytest

from kazusa_ai_chatbot.rag import memory_evidence_agent as memory_evidence_module
from kazusa_ai_chatbot.rag.conversation_evidence_agent import (
    ConversationEvidenceAgent,
)
from kazusa_ai_chatbot.rag.live_context_agent import LiveContextAgent
from kazusa_ai_chatbot.rag.memory_evidence_agent import MemoryEvidenceAgent
from kazusa_ai_chatbot.rag.person_context_agent import PersonContextAgent


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
        "current_timestamp": "2026-05-02T00:00:00+00:00",
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
    assert payload["primary_worker"] == "web_search_agent2"
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
    assert payload["primary_worker"] == "web_search_agent2"
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
            time_context={
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
            time_context={
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
            time_context={
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
async def test_live_context_current_local_time_requires_time_context() -> None:
    """Missing runtime time_context should stay unresolved without fallback."""
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
    assert result["result"]["missing_context"] == ["time_context"]
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
            time_context={
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
            time_context={
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
            time_context={
                "current_local_datetime": "2026-05-03 14:53",
                "current_local_weekday": "Sunday",
            }
        ),
    )

    assert result["resolved"] is True
    assert result["result"]["primary_worker"] == "web_search_agent2"
    assert len(web_worker.calls) == 1
    assert "fact_type=current_time" in web_worker.calls[0]["task"]
    assert memory_worker.calls == []
    assert conversation_worker.calls == []


@pytest.mark.asyncio
async def test_conversation_evidence_exact_phrase_uses_keyword_and_refs() -> None:
    """Exact phrase evidence should expose speaker, message, and URL refs."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker(
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
    search_worker = _FakeWorker({"resolved": True, "result": []})
    agent.keyword_agent = keyword_worker
    agent.search_agent = search_worker

    result = await agent.run(
        'Conversation-evidence: find who said "约定就是约定"',
        _base_context(current_platform_message_id="msg-current"),
    )

    assert result["resolved"] is True
    assert search_worker.calls == []
    assert len(keyword_worker.calls) == 1
    assert keyword_worker.calls[0]["context"]["exclude_current_question"] is True

    payload = result["result"]
    assert payload["primary_worker"] == "conversation_keyword_agent"
    assert payload["projection_payload"]["summaries"] == [
        "Tester at 2026-05-02 11:00: 约定就是约定, and here is https://example.test/post"
    ]
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
async def test_conversation_evidence_cjk_quotes_use_keyword_shortcut() -> None:
    """Quoted CJK phrases should not spend an LLM selector hop."""
    tasks = [
        "Conversation-evidence: find who said \u201c约定就是约定\u201d",
        "Conversation-evidence: find who said \u2018约定就是约定\u2019",
        "Conversation-evidence: find who said \u300c约定就是约定\u300d",
        "Conversation-evidence: find who said \u300e约定就是约定\u300f",
    ]

    for task in tasks:
        agent = ConversationEvidenceAgent()
        keyword_worker = _FakeWorker(
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
        search_worker = _FakeWorker({"resolved": True, "result": []})
        agent.keyword_agent = keyword_worker
        agent.search_agent = search_worker

        result = await agent.run(
            task,
            _base_context(current_platform_message_id="msg-current"),
        )

        assert result["resolved"] is True
        assert result["result"]["primary_worker"] == "conversation_keyword_agent"
        assert len(keyword_worker.calls) == 1
        assert search_worker.calls == []


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
    agent.keyword_agent = keyword_worker

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
    assert new_result["result"]["primary_worker"] == "conversation_keyword_agent"


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
    agent.keyword_agent = keyword_worker

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
    assert new_result["result"]["primary_worker"] == "conversation_keyword_agent"


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
async def test_conversation_evidence_rejects_active_agreement_intent() -> None:
    """Active episode agreement lookup belongs to Recall, not chat search."""
    agent = ConversationEvidenceAgent()
    keyword_worker = _FakeWorker({"resolved": True, "result": []})
    agent.keyword_agent = keyword_worker

    result = await agent.run(
        "Conversation-evidence: retrieve active agreement for today's appointment",
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
    agent.keyword_agent = keyword_worker
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
async def test_memory_evidence_exact_memory_name_uses_keyword() -> None:
    """Literal memory identifiers should use memory keyword evidence."""
    agent = MemoryEvidenceAgent()
    keyword_worker = _FakeWorker(
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
    search_worker = _FakeWorker({"resolved": True, "result": []})
    agent.keyword_agent = keyword_worker
    agent.search_agent = search_worker

    result = await agent.run(
        "Memory-evidence: exact memory_name active-character-official-address",
        _base_context(),
    )

    assert result["resolved"] is True
    assert len(keyword_worker.calls) == 1
    assert search_worker.calls == []
    assert result["result"]["primary_worker"] == "persistent_memory_keyword_agent"


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


def test_memory_evidence_selector_prompt_renders_scoped_worker_option() -> None:
    """The selector prompt should expose the scoped user-memory worker."""
    prompt = memory_evidence_module._SELECTOR_PROMPT

    assert "# Input Format" in prompt
    assert "# Output Format" in prompt
    assert "user_memory_evidence_agent" in prompt
    assert "current-user durable memory" in prompt


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
async def test_memory_evidence_old_setting_slot_uses_scoped_context() -> None:
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

    result = await agent.run(
        'Memory-evidence: retrieve durable evidence about “学姐抹茶冰淇淋店” setting',
        _base_context(
            original_query='请根据你和当前用户之间已经形成的私有连续性，回忆一下“学姐抹茶冰淇淋店”那个设定。',
        ),
    )

    assert result["resolved"] is True
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

    with caplog.at_level("DEBUG", logger="kazusa_ai_chatbot.rag.live_context_agent"):
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
