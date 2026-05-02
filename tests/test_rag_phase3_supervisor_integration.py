"""Deterministic integration checks for RAG2 capability slots."""

from __future__ import annotations

import json

import pytest

from kazusa_ai_chatbot.nodes import persona_supervisor2_rag_supervisor2 as supervisor2_module
from kazusa_ai_chatbot.rag.cache2_runtime import RAGCache2Runtime
from kazusa_ai_chatbot.rag.live_context_agent import LiveContextAgent

pytestmark = pytest.mark.asyncio


class _DummyResponse:
    """Small LangChain-like response wrapper for deterministic LLM tests."""

    def __init__(self, content: str) -> None:
        """Create a dummy response with model-compatible content."""
        self.content = content


class _InitializerLLM:
    """Static initializer fake that emits one approved slot."""

    def __init__(self, slot: str) -> None:
        """Store the slot returned by the fake initializer."""
        self.slot = slot

    async def ainvoke(self, _messages: list) -> _DummyResponse:
        """Return one JSON initializer payload."""
        payload = {"unknown_slots": [self.slot]}
        response = _DummyResponse(json.dumps(payload))
        return response


class _SummaryLLM:
    """Evaluator summarizer fake that preserves selected capability evidence."""

    async def ainvoke(self, messages: list) -> _DummyResponse:
        """Extract selected_summary from the evaluator payload."""
        payload = json.loads(messages[1].content)
        raw_result = payload["raw_result"]
        if isinstance(raw_result, dict):
            summary = str(raw_result.get("selected_summary", ""))
        else:
            summary = str(raw_result)
        response = _DummyResponse(summary)
        return response


class _FinalizerLLM:
    """Finalizer fake that echoes the first known-fact summary."""

    async def ainvoke(self, messages: list) -> _DummyResponse:
        """Return a compact final answer from known facts."""
        payload = json.loads(messages[1].content)
        known_facts = payload["known_facts"]
        summary = ""
        if known_facts:
            summary = str(known_facts[0]["summary"])
        response = _DummyResponse(f"final: {summary}")
        return response


class _FakeWorker:
    """Async worker test double that records helper-agent calls."""

    def __init__(self, result: dict) -> None:
        """Create a worker that always returns ``result``."""
        self.result = result
        self.calls: list[dict] = []

    async def run(
        self,
        task: str,
        context: dict,
        max_attempts: int = 3,
    ) -> dict:
        """Record the call and return the configured result."""
        self.calls.append(
            {
                "task": task,
                "context": context,
                "max_attempts": max_attempts,
            }
        )
        return_value = self.result
        return return_value


async def _noop_async(*args, **kwargs) -> None:
    """Accept scheduled persistent-cache writes without touching external DB."""
    del args, kwargs


def _prompt_context(query: str) -> dict:
    """Build prompt-safe current-message context for one integration case."""
    context = {
        "platform": "qq",
        "platform_channel_id": "rag-phase3-integration",
        "global_user_id": "user-1",
        "user_name": "Tester",
        "prompt_message_context": {
            "body_text": query,
            "mentions": [],
            "attachments": [],
            "addressed_to_global_user_ids": [],
            "broadcast": False,
        },
    }
    return context


async def _run_patched_live_context_case(
    monkeypatch,
    *,
    query: str,
    slot: str,
    web_text: str,
) -> tuple[dict, _FakeWorker]:
    """Run the RAG2 graph with a patched live-context web worker."""
    runtime = RAGCache2Runtime(max_entries=10)
    live_agent = LiveContextAgent()
    web_worker = _FakeWorker(
        {
            "resolved": True,
            "result": web_text,
            "attempts": 1,
            "cache": {
                "enabled": False,
                "hit": False,
                "cache_name": "",
                "reason": "patched_web_worker",
            },
        }
    )
    live_agent.web_agent = web_worker

    registry_entry = dict(
        supervisor2_module._RAG_SUPERVISOR_AGENT_REGISTRY["live_context_agent"]
    )
    registry_entry["agent"] = live_agent.run

    monkeypatch.setitem(
        supervisor2_module._RAG_SUPERVISOR_AGENT_REGISTRY,
        "live_context_agent",
        registry_entry,
    )
    monkeypatch.setattr(supervisor2_module, "get_rag_cache2_runtime", lambda: runtime)
    monkeypatch.setattr(supervisor2_module, "_initializer_llm", _InitializerLLM(slot))
    monkeypatch.setattr(supervisor2_module, "_evaluator_summarizer_llm", _SummaryLLM())
    monkeypatch.setattr(supervisor2_module, "_finalizer_llm", _FinalizerLLM())
    monkeypatch.setattr(supervisor2_module, "upsert_initializer_entry", _noop_async)
    monkeypatch.setattr(supervisor2_module, "record_initializer_hit", _noop_async)

    result = await supervisor2_module.call_rag_supervisor(
        original_query=query,
        character_name="<active character>",
        context=_prompt_context(query),
    )
    return_value = (result, web_worker)
    return return_value


async def test_supervisor_routes_explicit_weather_through_live_context(
    monkeypatch,
    caplog,
) -> None:
    """RAG2 should dispatch current weather through Live-context then web."""
    with caplog.at_level("DEBUG", logger="kazusa_ai_chatbot"):
        result, web_worker = await _run_patched_live_context_case(
            monkeypatch,
            query="What's the current temperature in Auckland?",
            slot="Live-context: answer current temperature for explicit location Auckland",
            web_text="Auckland is 17 C now. Source: https://weather.example/auckland",
        )

    assert result["unknown_slots"] == []
    assert result["loop_count"] == 1
    assert len(web_worker.calls) == 1
    assert "Auckland" in web_worker.calls[0]["task"]
    assert "17 C" in result["answer"]
    assert result["known_facts"][0]["agent"] == "live_context_agent"
    raw_result = result["known_facts"][0]["raw_result"]
    assert raw_result["primary_worker"] == "web_search_agent2"
    assert raw_result["projection_payload"]["url"] == "https://weather.example/auckland"
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
    assert not any("worker_payloads" in message for message in info_messages)
    assert any("worker_payloads" in message for message in debug_messages)


async def test_supervisor_routes_opening_status_through_live_context(monkeypatch) -> None:
    """RAG2 should handle opening status as live external context."""
    result, web_worker = await _run_patched_live_context_case(
        monkeypatch,
        query="Is Christchurch Adventure Park open right now?",
        slot=(
            "Live-context: answer current opening status for explicit target "
            "Christchurch Adventure Park"
        ),
        web_text=(
            "Christchurch Adventure Park is open now. "
            "Source: https://status.example/adventure-park"
        ),
    )

    assert result["unknown_slots"] == []
    assert result["loop_count"] == 1
    assert len(web_worker.calls) == 1
    assert "opening_status" in web_worker.calls[0]["task"]
    assert "Christchurch Adventure Park" in web_worker.calls[0]["task"]
    assert "open now" in result["answer"]
    assert result["known_facts"][0]["agent"] == "live_context_agent"
    raw_result = result["known_facts"][0]["raw_result"]
    assert raw_result["primary_worker"] == "web_search_agent2"
    assert raw_result["projection_payload"]["url"] == (
        "https://status.example/adventure-park"
    )


async def test_evaluator_summary_prompt_uses_compact_capability_payload(
    monkeypatch,
) -> None:
    """Evaluator summarization should not receive heavy raw worker payloads."""

    class _CaptureLLM:
        """Capture the evaluator prompt payload and return a small summary."""

        def __init__(self) -> None:
            """Initialize without captured payload."""
            self.payload: dict = {}

        async def ainvoke(self, messages: list) -> _DummyResponse:
            """Store the user payload sent to the summarizer LLM."""
            self.payload = json.loads(messages[1].content)
            response = _DummyResponse("compact summary")
            return response

    huge_text = "x" * 60000
    raw_result = {
        "capability": "person_context",
        "primary_worker": "user_profile_agent",
        "supporting_workers": [],
        "source_policy": "profile lookup",
        "selected_summary": "",
        "resolved_refs": [
            {
                "ref_type": "person",
                "role": "profile_owner",
                "global_user_id": "user-1",
                "display_name": "User",
            }
        ],
        "projection_payload": {
            "profile_kind": "third_party",
            "owner_global_user_id": "user-1",
            "summary": "",
            "profile": {
                "global_user_id": "user-1",
                "display_name": "User",
                "self_image": huge_text,
                "user_memory_context": {
                    "recent_shifts": [
                        {
                            "fact": huge_text,
                            "subjective_appraisal": huge_text,
                            "relationship_signal": huge_text,
                            "updated_at": "2026-05-02T00:00:00+00:00",
                        }
                        for _ in range(8)
                    ]
                },
                "_user_memory_units": [
                    {
                        "unit_type": "recent_shift",
                        "fact": huge_text,
                        "subjective_appraisal": huge_text,
                        "relationship_signal": huge_text,
                        "updated_at": "2026-05-02T00:00:00+00:00",
                    }
                    for _ in range(8)
                ],
            },
        },
        "worker_payloads": {
            "user_profile_agent": {
                "resolved": True,
                "result": huge_text,
            }
        },
        "evidence": [],
        "missing_context": [],
        "conflicts": [],
    }
    known_facts = [
        {
            "slot": "Conversation-evidence: identify speaker",
            "agent": "conversation_evidence_agent",
            "resolved": True,
            "summary": huge_text,
            "raw_result": raw_result,
            "attempts": 1,
        }
    ]
    capture_llm = _CaptureLLM()
    monkeypatch.setattr(
        supervisor2_module,
        "_evaluator_summarizer_llm",
        capture_llm,
    )

    summary = await supervisor2_module._summarize_agent_result(
        "Person-context: retrieve profile/impression for speaker found in slot 1",
        "person_context_agent",
        True,
        raw_result,
        known_facts,
    )

    rendered_payload = json.dumps(capture_llm.payload, ensure_ascii=False)
    assert summary == "compact summary"
    assert "worker_payloads" not in rendered_payload
    assert huge_text not in rendered_payload
    assert len(rendered_payload) < 20000
    profile = capture_llm.payload["raw_result"]["projection_payload"]["profile"]
    memory_context = profile["user_memory_context"]
    assert len(memory_context["recent_shifts"]) == 4

    delegate_context = supervisor2_module._build_delegate_context(
        {
            "context": {},
            "known_facts": known_facts,
            "original_query": "query",
            "current_slot": "Conversation-evidence: retrieve recent messages",
        },
        {"context": {}},
    )
    rendered_delegate_context = json.dumps(
        delegate_context,
        ensure_ascii=False,
    )
    assert "worker_payloads" not in rendered_delegate_context
    assert huge_text not in rendered_delegate_context
    assert len(rendered_delegate_context) < 20000
